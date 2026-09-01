import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from models.decoder_sr import HFBranch, LightWeightPyramidLaplacianSR, laplacian_hf, mask_dilate


class DecoderConditionedDataset(torch.utils.data.Dataset):
    """Pair the unchanged DiT-resolution sample with a high-resolution decoder input."""

    def __init__(self, low_dataset, high_dataset):
        if len(low_dataset) != len(high_dataset):
            raise ValueError("Low- and high-resolution datasets must have equal length.")
        self.low_dataset = low_dataset
        self.high_dataset = high_dataset

    def __len__(self):
        return len(self.low_dataset)

    def __getitem__(self, index):
        low = self.low_dataset[index]
        high = self.high_dataset[index]
        if len(low["video"]) != 2 or len(high["video"]) != 2:
            raise ValueError("Metadata video must contain [overexposed, underexposed].")
        low["decoder_video"] = high["video"]
        return low


def resize_center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def pil_pair_to_tensor(pair, height, width, device, dtype):
    tensors = [ToTensor()(resize_center_crop(image.convert("RGB"), height, width)) for image in pair]
    return torch.stack(tensors).to(device=device, dtype=dtype)


class HFConditionedWanVAE(nn.Module):
    """Frozen Wan VAE whose decoder receives the trained HF branch at four scales."""

    def __init__(self, base_vae, decoder_checkpoint):
        super().__init__()
        self.base_vae = base_vae
        self.hf_branch = HFBranch(sr_channels=256, hf_channels=64)
        checkpoint = torch.load(decoder_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("trainable_model", checkpoint)
        if "hf_branch" not in state:
            raise KeyError("Decoder checkpoint does not contain 'hf_branch'.")
        self.hf_branch.load_state_dict(state["hf_branch"], strict=True)
        base_parameter = next(
            (
                parameter
                for parameter in self.base_vae.parameters()
                if parameter.is_floating_point()
            ),
            None,
        )
        if base_parameter is None:
            raise RuntimeError(
                "The base Wan VAE has no floating-point parameters from "
                "which to infer decoder device and dtype."
            )
        self.hf_branch.to(
            device=base_parameter.device,
            dtype=base_parameter.dtype,
        )
        self.hf_branch.eval().requires_grad_(False)
        self.base_vae.eval().requires_grad_(False)
        self._condition = None

    @property
    def upsampling_factor(self):
        return self.base_vae.upsampling_factor

    @property
    def z_dim(self):
        return self.base_vae.z_dim

    @property
    def model(self):
        return self.base_vae.model

    def train(self, mode=True):
        super().train(mode)
        self.base_vae.eval()
        self.hf_branch.eval()
        return self

    def encode(self, *args, **kwargs):
        return self.base_vae.encode(*args, **kwargs)

    def set_condition(self, oe_hr, ue_hr, occ_mask):
        if oe_hr.shape != ue_hr.shape:
            raise ValueError("Decoder OE and UE must have the same shape.")
        self._condition = (oe_hr.detach().to("cpu"), ue_hr.detach().to("cpu"), occ_mask.detach().to("cpu"))

    def clear_condition(self):
        self._condition = None

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), hf_feat_cache=None):
        if hf_feat_cache is not None:
            return self.base_vae.decode(hidden_states, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, hf_feat_cache=hf_feat_cache)
        if self._condition is None:
            return self.base_vae.decode(hidden_states, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if tiled:
            raise ValueError("HF-conditioned VAE decoding does not support tiling.")
        if hidden_states.ndim != 5 or hidden_states.shape[2] != 1:
            raise ValueError("HF-conditioned Wan VAE currently supports one latent frame.")
        oe, ue, occ = self._condition
        lr_hw = (hidden_states.shape[-2] * self.upsampling_factor, hidden_states.shape[-1] * self.upsampling_factor)
        expected_hr = (lr_hw[0] * 3, lr_hw[1] * 3)
        if oe.shape[-2:] != expected_hr:
            raise ValueError(f"Decoder condition has size {oe.shape[-2:]}, expected {expected_hr} for latent {tuple(hidden_states.shape[-2:])}.")
        hf_dtype = next(self.hf_branch.parameters()).dtype
        oe = oe.to(device=device, dtype=hf_dtype)
        ue = ue.to(device=device, dtype=hf_dtype)
        occ = occ.to(device=device, dtype=hf_dtype)
        with torch.no_grad():
            vae_cache, _ = self.hf_branch(oe_hr=oe, ue_hr=ue, occ_mask=occ, latent_hw=hidden_states.shape[-2:], lr_hw=lr_hw)
        vae_cache = [
            feature.to(device=device, dtype=hidden_states.dtype)
            for feature in vae_cache
        ]
        return self.base_vae.decode(hidden_states, device=device, tiled=False, tile_size=tile_size, tile_stride=tile_stride, hf_feat_cache=vae_cache)


def build_arbitrary_sr_cache(hf_branch, oe, ue, occ_mask, target_hw):
    invalid = mask_dilate(occ_mask, radius=2)
    hf_rgb = laplacian_hf(oe) + laplacian_hf(ue) * (1 - invalid)
    hf_hr = hf_branch.hf_in(hf_rgb)
    mid_hw = (round(target_hw[0] * 2 / 3), round(target_hw[1] * 2 / 3))
    sr_2x = hf_branch.hf_in_sr_2x(F.interpolate(hf_hr, size=mid_hw, mode="bicubic", align_corners=False))
    sr_hr = hf_branch.hf_in_sr_hr(F.interpolate(hf_hr, size=target_hw, mode="bicubic", align_corners=False))
    return [sr_2x, sr_hr]


def run_arbitrary_sr(sr_model: LightWeightPyramidLaplacianSR, image_lr, hf_feat_cache, target_hw):
    x = sr_model.lr_in(image_lr)
    x = F.interpolate(x, size=hf_feat_cache[0].shape[-2:], mode="bicubic", align_corners=False)
    x = sr_model.up2(x + hf_feat_cache[0])
    x = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)
    x = sr_model.up_hr(x + hf_feat_cache[1])
    base = F.interpolate(image_lr, size=target_hw, mode="bicubic", align_corners=False)
    return base + sr_model.head(x), base
