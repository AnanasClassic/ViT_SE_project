#!/usr/bin/env python3
"""
ViT Fine-tuning on CIFAR-10
Training mini_vit architecture on CIFAR-10 dataset.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Sampler, random_split
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
import numpy as np
import random
import os
from datetime import datetime
from tqdm import tqdm

from src.models.mini_vit import MiniViT


class RepeatedAugmentationSampler(Sampler):
    def __init__(self, dataset, num_repeats=1, shuffle=True):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.num_samples = len(dataset) * num_repeats

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        indices = indices * self.num_repeats
        if self.shuffle:
            random.shuffle(indices)
        
        return iter(indices)

    def __len__(self):
        return self.num_samples


class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr = lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count < self.warmup_steps:
            lr = self.lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, beta=1.0):
    lam = np.random.beta(beta, beta) if beta > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    W = x.size(2)
    H = x.size(3)
    
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[..., bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (H * W)
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    num_batches = len(train_loader)

    pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    for batch_idx, (images, targets) in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        r = np.random.rand()
        if r < args.mix_prob and args.mixup_alpha == 0:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, args.cutmix_beta)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif r < args.mix_prob and args.mixup_alpha > 0:
            choice = np.random.rand()
            if choice < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, args.cutmix_beta)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, targets, args.mixup_alpha)
            
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = scheduler.step()
        
        with torch.no_grad():
            _, pred = outputs.max(1)
            correct = pred.eq(targets).sum().item()
            total_correct += correct
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)
        
        acc = 100.0 * correct / targets.size(0)
        
        # if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
        #     print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | Acc: {acc:.2f}% | LR: {lr:.6f}")

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%', 'LR': f'{lr:.6f}'})

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            _, pred = outputs.max(1)
            correct = pred.eq(targets).sum().item()
            
            total_loss += loss.item() * targets.size(0)
            total_correct += correct
            total_samples += targets.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT on CIFAR-10')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    
    # Model parameters
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')
    parser.add_argument('--img-size', type=int, default=32, help='Image size')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--channel', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=9, help='Number of layers')
    parser.add_argument('--heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp-ratio', type=int, default=2, help='MLP hidden dimension ratio')
    parser.add_argument('--num-cls', type=int, default=2, help='Number of CLS tokens')
    
    # Augmentation parameters
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='MixUp alpha')
    parser.add_argument('--cutmix-beta', type=float, default=1.0, help='CutMix beta')
    parser.add_argument('--mix-prob', type=float, default=0.5, help='Mix probability')
    parser.add_argument('--random-erasing-prob', type=float, default=0.25, help='Random erasing probability')
    parser.add_argument('--repeated-aug', type=int, default=3, help='Repeated augmentation')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--eval-epochs', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='training/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='Training log file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (f"ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_"
                f"ch{args.channel}_d{args.depth}_h{args.heads}_"
                f"cls{args.num_cls}_{timestamp}")
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    args.log_file = f"training_log_{run_name}.txt"
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Run name: {run_name}")
    print()
    
    hyperparams_str = "=" * 60 + "\nHyperparameters:\n" \
        + f"  Seed: {args.seed}\n" \
        + f"  Batch size: {args.batch_size}\n" \
        + f"  Epochs: {args.epochs}\n" \
        + f"  Warmup epochs: {args.warmup_epochs}\n" \
        + f"  Learning rate: {args.lr}\n" \
        + f"  Weight decay: {args.weight_decay}\n" \
        + f"  Label smoothing: {args.label_smoothing}\n" \
        + f"  MixUp alpha: {args.mixup_alpha}\n" \
        + f"  CutMix beta: {args.cutmix_beta}\n" \
        + f"  Mix probability: {args.mix_prob}\n" \
        + f"  Random erasing prob: {args.random_erasing_prob}\n" \
        + f"  Repeated augmentation: {args.repeated_aug}\n" \
        + "=" * 60 + "\n"
    
    print(hyperparams_str)
    with open(args.log_file, "w") as f:
        f.write(hyperparams_str)
    
    # Data loading
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.img_size, padding=4),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
        RandomErasing(p=args.random_erasing_prob, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_sampler = RepeatedAugmentationSampler(train_dataset, num_repeats=args.repeated_aug, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Dataset loaded:")
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()
    
    # Model
    model = MiniViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        n_classes=args.num_classes,
        emb_dim=args.channel,
        n_layers=args.depth,
        n_heads=args.heads,
        mlp_hidden_dim=args.channel * args.mlp_ratio,
        num_cls=args.num_cls,
    )
    
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_str = "=" * 60 + "\nModel Summary:\n" \
        + f"  Model: MiniViT with Multi-CLS\n" \
        + f"  Trainable parameters: {num_params:,}\n" \
        + f"  Image size: {args.img_size}x{args.img_size}\n" \
        + f"  Patch size: {args.patch_size}\n" \
        + f"  Dim: {args.channel}\n" \
        + f"  Depth: {args.depth}\n" \
        + f"  Heads: {args.heads}\n" \
        + f"  MLP ratio: {args.mlp_ratio}\n" \
        + f"  Num CLS tokens: {args.num_cls}\n" \
        + f"  Classifier input: {args.channel * args.num_cls}\n" \
        + "=" * 60 + "\n"
    print(model_str)
    with open(args.log_file, "a") as f:
        f.write(model_str)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps, args.lr)
    
    print(f"Training setup:")
    print(f"  Criterion: CrossEntropyLoss with label smoothing={args.label_smoothing}")
    print(f"  Optimizer: AdamW(lr={args.lr}, wd={args.weight_decay})")
    print(f"  Scheduler: CosineAnnealingWithWarmup")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  MixUp: {'Disabled' if args.mixup_alpha == 0 else f'Enabled (alpha={args.mixup_alpha})'}")
    print(f"  CutMix: {'Disabled' if args.cutmix_beta == 0 else f'Enabled (beta={args.cutmix_beta})'}")
    print()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_acc = 0
    
    print("=" * 60)
    print("Starting training...")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log file: {args.log_file}")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, epoch)
        
        if (epoch + 1) % args.eval_epochs == 0 or epoch == args.epochs - 1:
            test_loss, test_acc = validate(model, test_loader, criterion, device)
            
            out_str = (f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(out_str)
            
            with open(args.log_file, "a") as f:
                f.write(out_str + "\n")
            
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'args': vars(args),
                }, checkpoint_path)
                print(f"Best model saved! Accuracy: {best_acc:.2f}%")
    
    print()
    print("=" * 60)
    print(f"Training completed! Best test accuracy: {best_acc:.2f}%")
    print("=" * 60)
    print()
    
    with open(args.log_file, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Training completed! Best test accuracy: {best_acc:.2f}%\n")
        f.write(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()