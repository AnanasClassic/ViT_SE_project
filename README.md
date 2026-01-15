# Mini-ViT Implementation for CIFAR-10

A complete from-scratch implementation of Vision Transformer (ViT) architecture trained on CIFAR-10 dataset, achieving **93.5% test accuracy** on CPU.

## Task Requirements

The project implements a Mini-ViT model from scratch with the following mandatory requirements:

1. ✅ **Full manual implementation** of Vision Transformer components:
   - Patch embedding with convolutional projection
   - CLS token(s) and 2D sinusoidal positional embeddings
   - Multi-Head Self-Attention mechanism
   - Transformer encoder blocks
   - Classification head

2. ✅ **No pre-built transformer modules** - all attention and encoder logic implemented manually

3. ✅ **CPU training** with successful completion without errors

4. ✅ **Train loss convergence** - monotonic decrease over multiple epochs

5. ✅ **Val accuracy > 20%** (random baseline) - achieved **93.5%**

6. ✅ **Attention correctness verification** with logged tensor shapes and softmax normalization checks:
   - Q, K, V, A (attention weights), O (output) tensor shapes
   - Softmax normalization: max|sum−1| < 1e-4

7. ✅ **Best checkpoint saved** at `checkpoints/best.pt`

8. ✅ **Parameter count < 5M** - model has **674,890 parameters**

## Additional Features

Beyond the base requirements, this implementation includes:

### Advanced Architecture
- **Multi-CLS token design**: Supports multiple CLS tokens (default: 2) for enhanced representation learning
- **Flexible configuration**: Customizable depth, heads, embedding dimension, MLP ratio
- **LayerNorm pre-normalization**: Modern transformer architecture with pre-norm design

### Training Optimizations
- **CutMix augmentation**: Beta=1.0 with 50% probability for improved generalization
- **AutoAugment**: CIFAR10-specific policy
- **Random Erasing**: 25% probability
- **Repeated augmentation**: 3x sampling with different augmentations per epoch
- **Label smoothing**: 0.1
- **Cosine annealing with warmup**: 10-epoch warmup period
- **AdamW optimizer**: Weight decay 0.05
- **Batch size 512**: With gradient accumulation across epochs

### Engineering
- **Comprehensive logging**: Training progress, hyperparameters, and metrics tracked
- **Checkpoint management**: Auto-saves best model based on validation accuracy
- **Reproducible runs**: Seeded random states
- **Progress tracking**: tqdm integration for training visualization

## Results

**Best Model Performance:**
- Test Accuracy: **93.50%**
- Model Parameters: 674,890
- Training Duration: 250 epochs
- Architecture: 9 layers, 12 heads, 96-dim embeddings, 2 CLS tokens

Training configuration:
```
Batch size: 512
Learning rate: 0.002
Optimizer: AdamW (weight decay: 0.05)
Scheduler: Cosine annealing with 10-epoch warmup
Augmentation: CutMix + AutoAugment + Random Erasing + 3x repeated sampling
```

## Setup

### Requirements
- Python >= 3.10
- PyTorch with torchvision
- NumPy, tqdm

### Installation

Using `uv` (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install torch torchvision numpy tqdm
```

Or install from pyproject.toml:
```bash
pip install -e .
```

## Reproducing Experiments

### Quick Start - Best Model
Reproduce the best result (93.5% accuracy):
```bash
python finetune.py \
  --epochs 250 \
  --batch-size 512 \
  --lr 0.002 \
  --channel 96 \
  --depth 9 \
  --heads 12 \
  --num-cls 2 \
  --cutmix-beta 1.0 \
  --mix-prob 0.5 \
  --repeated-aug 3
```

### Basic Training
Minimal training run with default settings:
```bash
python finetune.py
```

### Custom Configuration
```bash
python finetune.py \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --channel 96 \
  --depth 6 \
  --heads 8 \
  --num-cls 1
```

### Available Arguments

**Training parameters:**
- `--batch-size`: Batch size (default: 256)
- `--epochs`: Number of training epochs (default: 100)
- `--warmup-epochs`: Warmup period (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: AdamW weight decay (default: 0.05)
- `--label-smoothing`: Label smoothing factor (default: 0.1)

**Model architecture:**
- `--patch-size`: Patch size (default: 4)
- `--img-size`: Input image size (default: 32)
- `--num-classes`: Number of output classes (default: 10)
- `--channel`: Embedding dimension (default: 96)
- `--depth`: Number of transformer layers (default: 9)
- `--heads`: Number of attention heads (default: 12)
- `--mlp-ratio`: MLP hidden dim ratio (default: 2)
- `--num-cls`: Number of CLS tokens (default: 2)

**Augmentation:**
- `--mixup-alpha`: MixUp alpha (default: 0.0)
- `--cutmix-beta`: CutMix beta (default: 1.0)
- `--mix-prob`: Probability of applying mix augmentation (default: 0.5)
- `--random-erasing-prob`: Random erasing probability (default: 0.25)
- `--repeated-aug`: Repeated augmentation factor (default: 3)

### Notebooks

Interactive exploration and experimentation:
- `finetune.ipynb`: Training experiments and visualization
- `inference.ipynb`: Model evaluation and prediction
- `test.ipynb`: Model testing and validation

## Project Structure

```
vit_se_project/
├── src/
│   └── models/
│       ├── attention.py      # Multi-Head Self-Attention implementation
│       ├── layers.py         # MLP and auxiliary layers
│       └── mini_vit.py       # Main ViT architecture
├── finetune.py              # Training script
├── finetune/
│   └── checkpoints/         # Saved model checkpoints
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── pyproject.toml           # Dependencies
└── README.md
```

## Architecture Details

### Patch Embedding
- Convolutional projection: 3 → 96 channels, 4×4 kernel, stride 4
- Produces 8×8 = 64 patches from 32×32 images

### Positional Encoding
- 2D sinusoidal encoding for spatial awareness
- Separate encodings for width and height dimensions
- Combined patch and CLS token positional embeddings

### Transformer Encoder
- 9 stacked encoder blocks
- Each block: LayerNorm → Multi-Head Attention → LayerNorm → MLP
- Residual connections throughout
- 12 attention heads per block
- MLP hidden dimension: 192 (2× embedding dim)

### Multi-Head Self-Attention
- Parallel attention heads: 12
- Head dimension: 8 (96 / 12)
- Scaled dot-product attention
- Dropout for regularization

### Classification Head
- Multi-CLS aggregation: Concatenates 2 CLS tokens
- Linear projection: 192 → 10 classes

## Citation

If you use this code in your research, please cite:
```bibtex
@software{mini_vit_cifar10,
  author = {Vlad},
  title = {Mini-ViT Implementation for CIFAR-10},
  year = {2026},
  url = {https://github.com/ananasclassic/vit_se_project}
}
```
