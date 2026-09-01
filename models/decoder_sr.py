from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def checkpoint_if_training(
    training: bool, function, *inputs: torch.Tensor
) -> torch.Tensor:
    """Checkpoint train-time activations while preserving the eager eval path."""
    if (
        training
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in inputs)
    ):
        return checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def mask_dilate(mask01: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask01
    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        mask01, kernel_size=kernel_size, stride=1, padding=radius
    )


def laplacian_hf(x: torch.Tensor, kind: str = "8") -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, got {tuple(x.shape)}")
    if kind == "4":
        kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    elif kind == "8":
        kernel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    else:
        raise ValueError("kind must be '4' or '8'")

    channels = x.shape[1]
    weight = torch.tensor(kernel, device=x.device, dtype=x.dtype)
    weight = weight.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x, weight, groups=channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(x)))
        return x + self.scale * residual


class ResStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DeformAlign(nn.Module):
    """The lightweight convolutional alignment block used by GuidedSR."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(feature))


class HFInjectDeform(nn.Module):
    def __init__(self, out_channels: int, hf_channels: int):
        super().__init__()
        self.align = DeformAlign(hf_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(hf_channels, out_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.proj(self.align(feature))


class HFBranch(nn.Module):
    """Extract high-frequency features for the Wan VAE and x3 SR decoder."""

    def __init__(self, sr_channels: int = 256, hf_channels: int = 64):
        super().__init__()
        self.hf_in = nn.Sequential(
            nn.Conv2d(3, hf_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResStack(hf_channels, 1),
        )

        self.hf_in_vae = nn.ModuleList(
            [
                HFInjectDeform(96, hf_channels),
                HFInjectDeform(192, 96),
                HFInjectDeform(192, 192),
                HFInjectDeform(384, 192),
            ]
        )
        self.hf_in_sr_2x = HFInjectDeform(sr_channels, hf_channels)
        self.hf_in_sr_hr = nn.Conv2d(hf_channels, sr_channels, 1)

    @staticmethod
    def _resize(feature: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            feature, size=size_hw, mode="bicubic", align_corners=False
        )

    def forward(
        self,
        oe_hr: torch.Tensor,
        ue_hr: torch.Tensor,
        occ_mask: torch.Tensor,
        latent_hw: tuple[int, int],
        lr_hw: tuple[int, int],
        gain_hf: float = 1.0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if oe_hr.shape != ue_hr.shape:
            raise ValueError("OE and UE must have the same shape after alignment.")
        if occ_mask.shape[0] != oe_hr.shape[0] or occ_mask.shape[-2:] != oe_hr.shape[-2:]:
            raise ValueError("Occlusion mask does not match the HR inputs.")

        with torch.no_grad():
            hf_oe = laplacian_hf(oe_hr)
            hf_ue = laplacian_hf(ue_hr)
            invalid = mask_dilate(occ_mask, radius=2)
            hf_rgb = (hf_oe + hf_ue * (1 - invalid)) * gain_hf

        hf_hr = self.hf_in(hf_rgb)

        lr_h, lr_w = lr_hw
        latent_h, latent_w = latent_hw
        expected_latent_hw = (lr_h // 8, lr_w // 8)
        if (latent_h, latent_w) != expected_latent_hw:
            raise ValueError(
                f"Wan VAE latent size {(latent_h, latent_w)} is inconsistent "
                f"with LR size {lr_hw}; expected {expected_latent_hw}."
            )

        feat_lr = self.hf_in_vae[0](self._resize(hf_hr, lr_hw))
        feat_half = self.hf_in_vae[1](
            self._resize(feat_lr, (lr_h // 2, lr_w // 2))
        )
        feat_quarter = self.hf_in_vae[2](
            self._resize(feat_half, (lr_h // 4, lr_w // 4))
        )
        feat_bottleneck = self.hf_in_vae[3](
            self._resize(feat_quarter, latent_hw)
        )
        vae_cache = [feat_bottleneck, feat_quarter, feat_half, feat_lr]

        hr_h, hr_w = oe_hr.shape[-2:]
        sr_2x_hw = (lr_h * 2, lr_w * 2)
        if (hr_h, hr_w) != (lr_h * 3, lr_w * 3):
            raise ValueError(
                f"The SR decoder requires exact x3 supervision, got LR={lr_hw}, "
                f"HR={(hr_h, hr_w)}."
            )
        sr_cache = [
            self.hf_in_sr_2x(self._resize(hf_hr, sr_2x_hw)),
            self.hf_in_sr_hr(hf_hr),
        ]
        return vae_cache, sr_cache


class LightWeightPyramidLaplacianSR(nn.Module):
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.lr_in = nn.Sequential(
            nn.Conv2d(3, feat_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResStack(feat_channels, 2),
        )
        self.up2 = ResStack(feat_channels, 1)
        self.up_hr = ResStack(feat_channels, 1)
        self.head = nn.Conv2d(feat_channels, 3, 3, padding=1)

    def forward(
        self, image_lr: torch.Tensor, hf_feat_cache: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(hf_feat_cache) != 2:
            raise ValueError("The x3 SR decoder expects 2x and HR HF features.")

        x = checkpoint_if_training(self.training, self.lr_in, image_lr)
        x2 = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        if x2.shape != hf_feat_cache[0].shape:
            raise ValueError(
                f"2x SR feature mismatch: {tuple(x2.shape)} vs "
                f"{tuple(hf_feat_cache[0].shape)}"
            )
        def up2_with_hf(
            feature: torch.Tensor, hf_feature: torch.Tensor
        ) -> torch.Tensor:
            return self.up2(feature + hf_feature)

        x2 = checkpoint_if_training(
            self.training, up2_with_hf, x2, hf_feat_cache[0]
        )

        target_hw = hf_feat_cache[1].shape[-2:]
        xh = F.interpolate(x2, size=target_hw, mode="bicubic", align_corners=False)
        if xh.shape != hf_feat_cache[1].shape:
            raise ValueError(
                f"HR SR feature mismatch: {tuple(xh.shape)} vs "
                f"{tuple(hf_feat_cache[1].shape)}"
            )
        def up_hr_with_hf(
            feature: torch.Tensor, hf_feature: torch.Tensor
        ) -> torch.Tensor:
            return self.up_hr(feature + hf_feature)

        xh = checkpoint_if_training(
            self.training, up_hr_with_hf, xh, hf_feat_cache[1]
        )

        base = F.interpolate(
            image_lr, size=target_hw, mode="bicubic", align_corners=False
        )
        output = base + self.head(xh)
        return output, base


class DecoderSRModel(nn.Module):
    def __init__(self, fidelity_vae):
        super().__init__()
        self.hf_branch = HFBranch(sr_channels=256, hf_channels=64)
        self.fidelity_vae = fidelity_vae
        self.light_weight_pyramid_laplacian_sr = LightWeightPyramidLaplacianSR(
            feat_channels=256
        )
        self.freeze_fidelity_vae()

    def freeze_fidelity_vae(self) -> None:
        self.fidelity_vae.eval()
        self.fidelity_vae.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.fidelity_vae.eval()
        return self

    def trainable_parameters(self):
        yield from self.hf_branch.parameters()
        yield from self.light_weight_pyramid_laplacian_sr.parameters()

    def forward(
        self,
        gt_latent: torch.Tensor,
        oe_hr: torch.Tensor,
        ue_hr: torch.Tensor,
        occ_mask: torch.Tensor,
        gain_hf: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hr_h, hr_w = oe_hr.shape[-2:]
        if hr_h % 3 or hr_w % 3:
            raise ValueError("HR dimensions must be divisible by three.")
        lr_hw = (hr_h // 3, hr_w // 3)

        vae_cache, sr_cache = self.hf_branch(
            oe_hr=oe_hr,
            ue_hr=ue_hr,
            occ_mask=occ_mask,
            latent_hw=gt_latent.shape[-2:],
            lr_hw=lr_hw,
            gain_hf=gain_hf,
        )

        def decode_with_hf(
            latent: torch.Tensor, *hf_features: torch.Tensor
        ) -> torch.Tensor:
            return self.fidelity_vae.decode(
                latent,
                device=latent.device,
                tiled=False,
                hf_feat_cache=list(hf_features),
            )

        decoded = checkpoint_if_training(
            self.training, decode_with_hf, gt_latent, *vae_cache
        )
        output_lr = ((decoded + 1.0) * 0.5).squeeze(2)
        output_hr, base = self.light_weight_pyramid_laplacian_sr(
            output_lr, sr_cache
        )
        return output_hr, output_lr, base
