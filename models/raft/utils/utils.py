import torch
import torch.nn.functional as F


class InputPadder:
    """Pad images so their spatial dimensions are divisible by eight."""

    def __init__(self, dims, mode="sintel"):
        height, width = dims[-2:]
        pad_height = (((height // 8) + 1) * 8 - height) % 8
        pad_width = (((width // 8) + 1) * 8 - width) % 8
        if mode == "sintel":
            self._pad = [
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ]
        else:
            self._pad = [
                pad_width // 2,
                pad_width - pad_width // 2,
                0,
                pad_height,
            ]

    def pad(self, *inputs):
        return [
            F.pad(value, self._pad, mode="replicate")
            for value in inputs
        ]

    def unpad(self, value):
        height, width = value.shape[-2:]
        top, bottom = self._pad[2], height - self._pad[3]
        left, right = self._pad[0], width - self._pad[1]
        return value[..., top:bottom, left:right]


def bilinear_sampler(image, coordinates, mode="bilinear", mask=False):
    height, width = image.shape[-2:]
    x_grid, y_grid = coordinates.split([1, 1], dim=-1)
    x_grid = 2 * x_grid / max(width - 1, 1) - 1
    y_grid = 2 * y_grid / max(height - 1, 1) - 1
    grid = torch.cat([x_grid, y_grid], dim=-1)
    sampled = F.grid_sample(
        image,
        grid,
        mode=mode,
        align_corners=True,
    )
    if not mask:
        return sampled
    valid = (
        (x_grid > -1)
        & (y_grid > -1)
        & (x_grid < 1)
        & (y_grid < 1)
    )
    return sampled, valid.float()


def coords_grid(batch, height, width, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coordinates = torch.stack([x, y], dim=0).float()
    return coordinates[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(
        flow,
        size=size,
        mode=mode,
        align_corners=True,
    )
