# DCGAN (PyTorch)

Minimal DCGAN implementation for 64x64 image generation using PyTorch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

MNIST (1-channel):

```bash
python /workspace/dcgan/train.py --dataset mnist --out_dir /workspace/outputs/dcgan_mnist --epochs 1 --max_steps 300
```

CIFAR10 (3-channel):

```bash
python /workspace/dcgan/train.py --dataset cifar10 --out_dir /workspace/outputs/dcgan_cifar10 --epochs 1 --max_steps 300 --batch_size 128
```

Sample images and checkpoints will be saved in the chosen `--out_dir`.

## Notes

- The architecture targets 64x64 images; datasets are resized accordingly.
- Outputs are saved with pixel values in [0, 1] for visualization.
