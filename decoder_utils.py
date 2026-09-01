import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from raft.raft import RAFT


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    pred_lap = F.conv2d(pred.mean(1, keepdim=True), kernel, padding=1)
    target_lap = F.conv2d(target.mean(1, keepdim=True), kernel, padding=1)
    return F.l1_loss(pred_lap, target_lap)


class HDRMetadataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path: str,
        source_prefix: str = "",
        target_prefix: str = "/mnt/oss/",
        max_pixels: int = 3840 * 2160,
        size_multiple: int = 24,
    ):
        super().__init__()
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive.")
        if size_multiple <= 0:
            raise ValueError("size_multiple must be positive.")
        if max_pixels < size_multiple * size_multiple:
            raise ValueError("max_pixels must fit at least one model-size block.")

        self.metadata_path = Path(metadata_path)
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        self.max_pixels = max_pixels
        self.size_multiple = size_multiple
        self.to_tensor = ToTensor()
        self.samples = self._load_metadata()

    def _rewrite_path(self, value, key: str, index: int) -> Path:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    f"Sample {index} key '{key}' must be a path or a one-item list."
                )
            value = value[0]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Sample {index} key '{key}' is not a valid path.")
        if self.source_prefix and value.startswith(self.source_prefix):
            value = self.target_prefix + value[len(self.source_prefix):]
        return Path(value)

    def _parse_sample(self, sample: dict, index: int) -> dict:
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} in metadata is not an object.")
        if "gt" not in sample:
            raise KeyError(f"Sample {index} has no 'gt' path.")

        if "oe" in sample and "ue" in sample:
            oe_value, ue_value = sample["oe"], sample["ue"]
        else:
            video = sample.get("video")
            if not isinstance(video, (list, tuple)) or len(video) < 2:
                raise KeyError(
                    f"Sample {index} must contain 'oe'/'ue' or a two-item 'video' list."
                )
            oe_value, ue_value = video[0], video[1]

        paths = {
            "gt": self._rewrite_path(sample["gt"], "gt", index),
            "oe": self._rewrite_path(oe_value, "oe", index),
            "ue": self._rewrite_path(ue_value, "ue", index),
        }
        name = sample.get("name") or sample.get("id") or paths["gt"].stem
        return {**paths, "name": str(name)}

    def _load_metadata(self) -> list[dict]:
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Training metadata not found: {self.metadata_path}")
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Training metadata is not valid JSON: {self.metadata_path}"
            ) from error
        if isinstance(metadata, dict):
            metadata = metadata.get("data") or metadata.get("samples")
        if not isinstance(metadata, list) or not metadata:
            raise ValueError(
                "Training metadata must be a non-empty JSON list (or contain "
                "a non-empty 'data'/'samples' list)."
            )
        return [
            self._parse_sample(sample, index)
            for index, sample in enumerate(metadata)
        ]

    def _constrain_multiple_size(
        self, target_height: int, target_width: int
    ) -> tuple[int, int]:
        for _ in range(2):
            if target_height * target_width <= self.max_pixels:
                break
            if target_height >= target_width and target_height > self.size_multiple:
                target_height = max(
                    self.size_multiple,
                    self.max_pixels // target_width
                    // self.size_multiple
                    * self.size_multiple,
                )
            elif target_width > self.size_multiple:
                target_width = max(
                    self.size_multiple,
                    self.max_pixels // target_height
                    // self.size_multiple
                    * self.size_multiple,
                )
        return target_height, target_width

    @staticmethod
    def _crop_and_pad(
        images: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        height, width = images.shape[-2:]
        crop_height = min(height, target_height)
        crop_width = min(width, target_width)
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        images = images[
            :,
            :,
            top:top + crop_height,
            left:left + crop_width,
        ]
        pad_h = target_height - crop_height
        pad_w = target_width - crop_width
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="replicate")
        return images

    def _resize_and_align(self, images: torch.Tensor) -> torch.Tensor:
        _, _, height, width = images.shape
        if height * width > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (height * width))
            scaled_height = max(1, round(height * scale))
            scaled_width = max(1, round(width * scale))
            images = F.interpolate(
                images,
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            height, width = scaled_height, scaled_width
            target_height = max(
                self.size_multiple,
                height // self.size_multiple * self.size_multiple,
            )
            target_width = max(
                self.size_multiple,
                width // self.size_multiple * self.size_multiple,
            )
        else:
            target_height = (
                (height + self.size_multiple - 1)
                // self.size_multiple
                * self.size_multiple
            )
            target_width = (
                (width + self.size_multiple - 1)
                // self.size_multiple
                * self.size_multiple
            )
            if target_height * target_width > self.max_pixels:
                target_height = max(
                    self.size_multiple,
                    height // self.size_multiple * self.size_multiple,
                )
                target_width = max(
                    self.size_multiple,
                    width // self.size_multiple * self.size_multiple,
                )

        target_height, target_width = self._constrain_multiple_size(
            target_height, target_width
        )
        return self._crop_and_pad(images, target_height, target_width)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tensors = []
        sizes = []
        for key in ("gt", "oe", "ue"):
            try:
                with Image.open(sample[key]) as image:
                    image = image.convert("RGB")
                    sizes.append(image.size)
                    tensors.append(self.to_tensor(image))
            except (FileNotFoundError, OSError) as error:
                raise RuntimeError(
                    f"Failed to load sample '{sample['name']}' {key}: {sample[key]}"
                ) from error
        if len(set(sizes)) != 1:
            raise ValueError(
                f"Sample '{sample['name']}' is not spatially aligned: {sizes}"
            )
        images = self._resize_and_align(torch.stack(tensors))
        return {
            "name": sample["name"],
            "gt": images[0],
            "oe": images[1],
            "ue": images[2],
        }


class FlowAligner(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.flow_aligner = RAFT()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = OrderedDict(
            (key.replace("module.", ""), value)
            for key, value in state_dict.items()
        )
        self.flow_aligner.load_state_dict(state_dict)
        self.eval().requires_grad_(False)

    @staticmethod
    def _backward_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=0).float()
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
        sample_grid = grid + flow
        sample_grid[:, 0] = (
            2.0 * sample_grid[:, 0].clone() / max(width - 1, 1) - 1.0
        )
        sample_grid[:, 1] = (
            2.0 * sample_grid[:, 1].clone() / max(height - 1, 1) - 1.0
        )
        return F.grid_sample(
            x,
            sample_grid.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _forward_backward_consistency_check(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        alpha: float = 0.01,
        beta: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_magnitude = torch.norm(forward_flow, dim=1) + torch.norm(
            backward_flow, dim=1
        )
        warped_backward = self._backward_warp(backward_flow, forward_flow)
        warped_forward = self._backward_warp(forward_flow, backward_flow)
        forward_difference = torch.norm(forward_flow + warped_backward, dim=1)
        backward_difference = torch.norm(backward_flow + warped_forward, dim=1)
        threshold = alpha * flow_magnitude + beta
        return (
            (forward_difference > threshold).float(),
            (backward_difference > threshold).float(),
        )

    @staticmethod
    def _histogram_table(source_hist: torch.Tensor, target_hist: torch.Tensor):
        table = torch.zeros(256, device=source_hist.device)
        target_remaining = target_hist.clone()
        target_bin = 0
        for source_bin in range(256):
            if source_hist[source_bin] == 0:
                table[source_bin] = -1
                continue
            pixels_left = source_hist[source_bin]
            weighted_sum = source_hist.new_zeros(())
            while pixels_left > 0:
                take = torch.minimum(pixels_left, target_remaining[target_bin])
                weighted_sum = weighted_sum + take * target_bin
                pixels_left = pixels_left - take
                target_remaining[target_bin] = target_remaining[target_bin] - take
                if target_remaining[target_bin] == 0 and target_bin < 255:
                    target_bin += 1
            table[source_bin] = torch.round(
                weighted_sum / source_hist[source_bin]
            )
        return table.long()

    def _match_histogram(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        output = []
        for source_image, target_image in zip(source, target):
            channels = []
            for source_channel, target_channel in zip(source_image, target_image):
                source_index = (source_channel * 255).round().long().clamp(0, 255)
                target_index = (target_channel * 255).round().long().clamp(0, 255)
                source_hist = torch.bincount(
                    source_index.flatten(), minlength=256
                ).float()
                target_hist = torch.bincount(
                    target_index.flatten(), minlength=256
                ).float()
                table = self._histogram_table(source_hist, target_hist)
                channels.append(table[source_index].to(source.dtype) / 255.0)
            output.append(torch.stack(channels))
        return torch.stack(output).clamp(0, 1)

    def forward(
        self, ue: torch.Tensor, target: torch.Tensor, scale_factor: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ue.ndim != 4 or target.ndim != 4:
            raise ValueError("FlowAligner expects BCHW image tensors.")
        if ue.shape != target.shape:
            raise ValueError(
                "UE and target must have identical shapes, got "
                f"{tuple(ue.shape)} and {tuple(target.shape)}."
            )
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError("scale_factor must be a positive integer.")

        output_dtype = ue.dtype
        ue = ue.float()
        target = target.float()
        original_height, original_width = ue.shape[-2:]
        ue_origin = ue
        ue_matched = self._match_histogram(ue, target)

        size_multiple = 8 * scale_factor
        pad_height = (-original_height) % size_multiple
        pad_width = (-original_width) % size_multiple
        if pad_height or pad_width:
            padding = (0, pad_width, 0, pad_height)
            ue_origin = F.pad(ue_origin, padding, mode="replicate")
            ue_matched = F.pad(ue_matched, padding, mode="replicate")
            target = F.pad(target, padding, mode="replicate")

        padded_height, padded_width = target.shape[-2:]
        flow_height = padded_height // scale_factor
        flow_width = padded_width // scale_factor
        target_lr = F.interpolate(
            target,
            size=(flow_height, flow_width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        ue_lr = F.interpolate(
            ue_matched,
            size=(flow_height, flow_width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)

        _, ue_to_oe_lr = self.flow_aligner(
            ue_lr * 2 - 1, target_lr * 2 - 1, iters=20, test_mode=True
        )
        _, oe_to_ue_lr = self.flow_aligner(
            target_lr * 2 - 1, ue_lr * 2 - 1, iters=20, test_mode=True
        )
        ue_to_oe = F.interpolate(
            ue_to_oe_lr,
            size=(padded_height, padded_width),
            mode="bicubic",
            align_corners=False,
        ) * scale_factor
        oe_to_ue = F.interpolate(
            oe_to_ue_lr,
            size=(padded_height, padded_width),
            mode="bicubic",
            align_corners=False,
        ) * scale_factor

        aligned_ue = self._backward_warp(ue_origin, oe_to_ue)
        _, occ_mask = self._forward_backward_consistency_check(
            ue_to_oe, oe_to_ue
        )
        occ_mask = occ_mask.unsqueeze(1)
        aligned_ue = aligned_ue[..., :original_height, :original_width]
        occ_mask = occ_mask[..., :original_height, :original_width]
        return aligned_ue.to(output_dtype), occ_mask.to(output_dtype)
