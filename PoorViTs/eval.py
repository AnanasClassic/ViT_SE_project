import argparse
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import gin

from utils.dataloader import datainfo, dataload
from models.build_model import create_model
from utils.training_functions import accuracy

MODELS = ['vit', 'swin', 'cait', 'mini_vit', 'mhc_vit', 'none']


class NullLogger:
    def debug(self, _msg):
        pass


def build_device(no_cuda):
    if (not no_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    if (not no_cuda) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def evaluate(val_loader, model, criterion, device):
    model.eval()
    loss_val = 0.0
    acc1_val = 0.0
    n = 0
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            if device.type != "cpu":
                images = images.to(device)
                target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            acc = accuracy(output, target, (1,))
            acc1 = acc[0]
            batch_size = images.size(0)
            n += batch_size
            loss_val += float(loss.item() * batch_size)
            acc1_val += float(acc1[0] * batch_size)
    return loss_val / n, acc1_val / n


def main():
    parser = argparse.ArgumentParser(description="PoorViTs evaluation script")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint")
    parser.add_argument("--datapath", default="./data", type=str, help="Dataset path")
    parser.add_argument("--dataset", default="CIFAR10", choices=[
        "CIFAR10", "CIFAR100", "Tiny-Imagenet", "SVHN", "CINIC"
    ], type=str)
    parser.add_argument("-b", "--batch_size", default=256, type=int)
    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument("--arch", type=str, default="vit", choices=MODELS)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--vit_mlp_ratio", default=2, type=int)
    parser.add_argument("--sd", default=0.1, type=float)
    parser.add_argument("--sin_pos", action="store_true", default=False)
    parser.add_argument("--channel", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--gin", nargs="+", type=str, default=[])
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA/MPS")
    args = parser.parse_args()

    gin.parse_config(args.gin)
    device = build_device(args.no_cuda)

    logger = NullLogger()
    data_info = datainfo(logger, args)
    normalize = [transforms.Normalize(*data_info["stat"])]
    augmentations = transforms.Compose([transforms.ToTensor(), *normalize])
    _, val_dataset = dataload(args, augmentations, normalize, data_info)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(data_info["img_size"], data_info["n_classes"], args)
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    val_loss, val_acc1 = evaluate(val_loader, model, criterion, device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"val_loss: {val_loss:.4f}")
    logging.info(f"val_acc1: {val_acc1:.2f}")


if __name__ == "__main__":
    main()
