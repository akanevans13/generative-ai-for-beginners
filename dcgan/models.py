import math
from typing import Tuple

import torch
from torch import nn


def initialize_dcgan_weights(module: nn.Module) -> None:
    """Apply DCGAN weight initialization: normal(0, 0.02) for conv/convT and batchnorm.

    This mirrors the initialization used in the DCGAN paper/tutorial and tends
    to improve training stability.
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


class Generator(nn.Module):
    """DCGAN Generator for 64x64 images.

    Assumptions:
    - Output image size is 64x64.
    - Output value range is [-1, 1] via Tanh at the output.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_num_feature_maps: int = 64,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Input Z goes into a transposed conv that creates a 4x4 feature map.
        # Then we upsample 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64.
        self.net = nn.Sequential(
            # Input Z: (N, latent_dim, 1, 1) -> (N, 8*F, 4, 4)
            nn.ConvTranspose2d(latent_dim, base_num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 8),
            nn.ReLU(True),

            # (N, 8*F, 4, 4) -> (N, 4*F, 8, 8)
            nn.ConvTranspose2d(base_num_feature_maps * 8, base_num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 4),
            nn.ReLU(True),

            # (N, 4*F, 8, 8) -> (N, 2*F, 16, 16)
            nn.ConvTranspose2d(base_num_feature_maps * 4, base_num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 2),
            nn.ReLU(True),

            # (N, 2*F, 16, 16) -> (N, F, 32, 32)
            nn.ConvTranspose2d(base_num_feature_maps * 2, base_num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps),
            nn.ReLU(True),

            # (N, F, 32, 32) -> (N, C, 64, 64)
            nn.ConvTranspose2d(base_num_feature_maps, output_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(initialize_dcgan_weights)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.dim() == 2:
            noise = noise.unsqueeze(-1).unsqueeze(-1)
        return self.net(noise)


class Discriminator(nn.Module):
    """DCGAN Discriminator for 64x64 images.

    Produces a single logit per image (use with BCEWithLogitsLoss).
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_num_feature_maps: int = 64,
    ) -> None:
        super().__init__()

        # (N, C, 64, 64) -> (N, F, 32, 32)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, F, 32, 32) -> (N, 2*F, 16, 16)
            nn.Conv2d(base_num_feature_maps, base_num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 2*F, 16, 16) -> (N, 4*F, 8, 8)
            nn.Conv2d(base_num_feature_maps * 2, base_num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 4*F, 8, 8) -> (N, 8*F, 4, 4)
            nn.Conv2d(base_num_feature_maps * 4, base_num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_num_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 8*F, 4, 4) -> (N, 1, 1, 1)
            nn.Conv2d(base_num_feature_maps * 8, 1, 4, 1, 0, bias=False),
        )

        self.apply(initialize_dcgan_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.net(images)
        return logits.view(-1)