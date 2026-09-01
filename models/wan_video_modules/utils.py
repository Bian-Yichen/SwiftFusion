from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.raft.raft import RAFT
from models.raft.utils.utils import InputPadder


class FlowAligner(nn.Module):
    """Align the underexposed image to the overexposed reference."""

    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RAFT checkpoint not found: {checkpoint_path}"
            )

        self.raft = RAFT()
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in state_dict.items()
        )
        self.raft.load_state_dict(state_dict)
        self.raft.requires_grad_(False)
        self.raft.eval()

    def train(self, mode=True):
        super().train(mode)
        self.raft.eval()
        return self

    @staticmethod
    def _histogram_table(source_hist, target_hist):
        source_hist = source_hist.float()
        target_hist = target_hist.float().clone()
        table = torch.zeros(
            256,
            device=source_hist.device,
            dtype=torch.float32,
        )
        target_bin = 0
        for source_bin in range(256):
            source_count = source_hist[source_bin]
            if source_count == 0:
                table[source_bin] = source_bin
                continue
            remaining = source_count
            weighted_sum = torch.zeros_like(remaining)
            while remaining > 0 and target_bin < 256:
                take = torch.minimum(
                    remaining,
                    target_hist[target_bin],
                )
                weighted_sum += take * target_bin
                remaining -= take
                target_hist[target_bin] -= take
                if target_hist[target_bin] <= 0:
                    target_bin += 1
            table[source_bin] = weighted_sum / source_count
        return table.round()

    @classmethod
    def match_histogram(cls, source, target):
        if source.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("FlowAligner currently requires batch size 1.")
        source_dtype = source.dtype
        source_255 = (source.float() * 255).round().clamp(0, 255)
        target_255 = (target.float() * 255).round().clamp(0, 255)
        output = torch.empty_like(source_255)
        for channel in range(source.shape[1]):
            source_channel = source_255[0, channel]
            target_channel = target_255[0, channel]
            source_hist = torch.histc(
                source_channel,
                bins=256,
                min=0,
                max=255,
            )
            target_hist = torch.histc(
                target_channel,
                bins=256,
                min=0,
                max=255,
            )
            table = cls._histogram_table(
                source_hist,
                target_hist,
            )
            lower = source_channel.floor().long()
            upper = (lower + 1).clamp(max=255)
            fraction = source_channel - lower
            output[0, channel] = (
                table[lower] * (1 - fraction)
                + table[upper] * fraction
            )
        return (output / 255).clamp(0, 1).to(source_dtype)

    @staticmethod
    def estimate_homography(
        flow,
        stride=4,
        ransac_threshold=2.0,
        max_iterations=2000,
        confidence=0.999,
        max_points=2000,
    ):
        batch, _, height, width = flow.shape
        homographies = []
        for batch_id in range(batch):
            flow_np = flow[batch_id].float().cpu().numpy()
            grid_y, grid_x = np.meshgrid(
                np.arange(0, height, stride, dtype=np.int32),
                np.arange(0, width, stride, dtype=np.int32),
                indexing="ij",
            )
            x = grid_x.reshape(-1)
            y = grid_y.reshape(-1)
            source = np.stack([x, y], axis=1).astype(np.float32)
            target = np.stack(
                [
                    x + flow_np[0, grid_y, grid_x].reshape(-1),
                    y + flow_np[1, grid_y, grid_x].reshape(-1),
                ],
                axis=1,
            ).astype(np.float32)
            if source.shape[0] > max_points:
                indices = np.linspace(
                    0,
                    source.shape[0] - 1,
                    max_points,
                    dtype=np.int64,
                )
                source = source[indices]
                target = target[indices]
            homography, _ = cv2.findHomography(
                source,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=max_iterations,
                confidence=confidence,
            )
            if homography is None:
                homography = np.eye(3, dtype=np.float32)
            homographies.append(
                torch.from_numpy(homography).float()
            )
        return torch.stack(homographies).to(flow.device)

    @staticmethod
    def warp(image, homography):
        batch, _, height, width = image.shape
        y, x = torch.meshgrid(
            torch.arange(
                height,
                device=image.device,
                dtype=torch.float32,
            ),
            torch.arange(
                width,
                device=image.device,
                dtype=torch.float32,
            ),
            indexing="ij",
        )
        points = torch.stack(
            [x, y, torch.ones_like(x)],
            dim=0,
        ).reshape(3, -1)
        points = points.unsqueeze(0).expand(batch, -1, -1)
        source_points = torch.bmm(
            torch.linalg.inv(homography.float()),
            points,
        )
        source_x = source_points[:, 0] / (
            source_points[:, 2] + 1e-8
        )
        source_y = source_points[:, 1] / (
            source_points[:, 2] + 1e-8
        )
        grid = torch.stack(
            [
                2 * source_x / max(width - 1, 1) - 1,
                2 * source_y / max(height - 1, 1) - 1,
            ],
            dim=-1,
        ).reshape(batch, height, width, 2)
        return F.grid_sample(
            image.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(image.dtype)

    @torch.no_grad()
    def forward(
        self,
        underexposed,
        overexposed,
        scale_factor=4.0,
    ):
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive.")
        if underexposed.shape != overexposed.shape:
            raise ValueError(
                "Underexposed and overexposed inputs must have the "
                f"same shape, got {tuple(underexposed.shape)} and "
                f"{tuple(overexposed.shape)}."
            )

        underexposed = underexposed.float()
        overexposed = overexposed.float()
        matched = self.match_histogram(
            underexposed,
            overexposed,
        )

        height, width = underexposed.shape[-2:]
        flow_height = max(8, int(height / scale_factor))
        flow_width = max(8, int(width / scale_factor))
        scale_y = height / flow_height
        scale_x = width / flow_width
        matched_low = F.interpolate(
            matched,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=False,
        )
        overexposed_low = F.interpolate(
            overexposed,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=False,
        )

        padder = InputPadder(matched_low.shape)
        matched_padded, overexposed_padded = padder.pad(matched_low, overexposed_low)
        _, flow_padded = self.raft(
            matched_padded * 2 - 1,
            overexposed_padded * 2 - 1,
            iters=20,
            test_mode=True,
        )
        flow_low = padder.unpad(flow_padded)
        flow = F.interpolate(
            flow_low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flow_scale = torch.tensor(
            [scale_x, scale_y],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 2, 1, 1)

        flow = flow * flow_scale
        homography = self.estimate_homography(flow,)

        return self.warp(underexposed, homography).clamp(0, 1)
