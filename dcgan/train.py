import argparse
import json
import os
import random
from datetime import datetime
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

from models import Generator, Discriminator


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset_name: str,
    data_dir: str,
    image_size: int,
    channels: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    if dataset_name.lower() == "mnist":
        transform_ops = []
        # MNIST images are 1-channel; up-convert if channels == 3
        if channels == 3:
            transform_ops.append(transforms.Lambda(lambda img: img.convert("RGB")))
        elif channels != 1:
            raise ValueError("MNIST supports channels=1 or channels=3 when converting to RGB.")
        transform_ops.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ])
        dataset = datasets.MNIST(root=data_dir, train=True, transform=transforms.Compose(transform_ops), download=True)
    elif dataset_name.lower() == "cifar10":
        if channels != 3:
            raise ValueError("CIFAR10 requires channels=3.")
        transform_ops = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.Compose(transform_ops), download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from: mnist, cifar10")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader, channels


def save_config(out_dir: str, args: argparse.Namespace) -> None:
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DCGAN on MNIST or CIFAR10")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="/workspace/data", help="Directory to store/fetch the dataset")

    parser.add_argument("--out_dir", type=str, default="/workspace/outputs/dcgan", help="Output directory for samples and checkpoints")
    parser.add_argument("--image_size", type=int, default=64, help="Target image resolution (DCGAN expects 64)")
    parser.add_argument("--channels", type=int, default=None, help="Number of image channels (auto by dataset if None)")

    parser.add_argument("--latent_dim", type=int, default=128, help="Latent vector dimension")
    parser.add_argument("--g_features", type=int, default=64, help="Generator base feature maps")
    parser.add_argument("--d_features", type=int, default=64, help="Discriminator base feature maps")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")

    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--sample_every", type=int, default=200, help="Steps between sample image grids")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Epochs between checkpoints")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional limit on total training steps for a quick smoke test")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_config(args.out_dir, args)

    set_seed(args.seed)
    cudnn.benchmark = True

    # Resolve channels if not provided
    if args.channels is None:
        args.channels = 1 if args.dataset.lower() == "mnist" else 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, channels = build_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        image_size=args.image_size,
        channels=args.channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generator = Generator(latent_dim=args.latent_dim, base_num_feature_maps=args.g_features, output_channels=channels).to(device)
    discriminator = Discriminator(input_channels=channels, base_num_feature_maps=args.d_features).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for real_images, _ in progress:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            optimizer_d.zero_grad(set_to_none=True)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_labels)

            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_logits, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator: maximize log(D(G(z))) => minimize BCEWithLogitsLoss(fake_logits, 1)
            optimizer_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake_images)
            loss_g = criterion(fake_logits_for_g, real_labels)
            loss_g.backward()
            optimizer_g.step()

            progress.set_postfix({
                "loss_d": f"{loss_d.item():.3f}",
                "loss_g": f"{loss_g.item():.3f}",
            })

            # Save sample grid periodically
            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    samples = generator(fixed_noise).detach().cpu()
                    samples = (samples + 1.0) / 2.0
                    samples.clamp_(0.0, 1.0)
                grid_path = os.path.join(args.out_dir, f"samples_step_{global_step:07d}.png")
                vutils.save_image(samples, grid_path, nrow=8)

            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        # End epoch: checkpoint
        if epoch % args.checkpoint_every == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final sample
    with torch.no_grad():
        samples = generator(fixed_noise).detach().cpu()
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)
    grid_path = os.path.join(args.out_dir, f"samples_final.png")
    vutils.save_image(samples, grid_path, nrow=8)


if __name__ == "__main__":
    main()