import argparse
import random
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from decoder_utils import FlowAligner, HDRMetadataDataset, laplacian_loss
from models.decoder_model_pool import load_decoder_wan_vae
from models.decoder_sr import DecoderSRModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the HF-guided Wan VAE and x3 SR decoder without DiT."
    )
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--raft_checkpoint", required=True)
    parser.add_argument("--metadata_path", default="train_metadata.json")
    parser.add_argument(
        "--path_prefix_from", default=""
    )
    parser.add_argument("--path_prefix_to", default="/mnt/oss/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_pixels", type=int, default=3840 * 2160)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gain_hf", type=float, default=1.0)
    parser.add_argument("--detect_anomaly", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fidelity_vae(checkpoint_path: str, device, dtype):
    vae = load_decoder_wan_vae(checkpoint_path, device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae


def trainable_state_dict(model: DecoderSRModel):
    return {
        "hf_branch": model.hf_branch.state_dict(),
        "light_weight_pyramid_laplacian_sr": (
            model.light_weight_pyramid_laplacian_sr.state_dict()
        ),
    }


def load_trainable_state_dict(model: DecoderSRModel, state_dict):
    model.hf_branch.load_state_dict(state_dict["hf_branch"])
    model.light_weight_pyramid_laplacian_sr.load_state_dict(
        state_dict["light_weight_pyramid_laplacian_sr"]
    )


def save_checkpoint(
    path: Path,
    model: DecoderSRModel,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    args,
):
    torch.save(
        {
            "trainable_model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"], checkpoint["global_step"]


def verify_parameter_partition(model: DecoderSRModel, optimizer):
    vae_ids = {id(parameter) for parameter in model.fidelity_vae.parameters()}
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    overlap = vae_ids & optimized_ids
    if overlap:
        raise RuntimeError("Frozen Wan VAE parameters leaked into the optimizer.")
    if any(parameter.requires_grad for parameter in model.fidelity_vae.parameters()):
        raise RuntimeError("Wan VAE contains unexpectedly trainable parameters.")


def verify_frozen_vae_gradients(model: DecoderSRModel):
    leaked = [
        name
        for name, parameter in model.fidelity_vae.named_parameters()
        if parameter.grad is not None
    ]
    if leaked:
        raise RuntimeError(f"Frozen Wan VAE received gradients: {leaked[:5]}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Decoder training requires a CUDA GPU.")
    if args.batch_size != 1:
        raise ValueError(
            "Variable-resolution decoder training requires --batch_size 1."
        )

    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HDRMetadataDataset(
        metadata_path=args.metadata_path,
        source_prefix=args.path_prefix_from,
        target_prefix=args.path_prefix_to,
        max_pixels=args.max_pixels,
        size_multiple=24,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    fidelity_vae = load_fidelity_vae(args.vae_checkpoint, device, dtype)
    model = DecoderSRModel(fidelity_vae).to(device=device, dtype=dtype)
    aligner = FlowAligner(args.raft_checkpoint).to(device=device)
    perceptual_loss = lpips.LPIPS(net="vgg").to(
        device=device, dtype=dtype
    )

    model.train()
    aligner.eval().requires_grad_(False)
    perceptual_loss.eval().requires_grad_(False)

    trainable_parameters = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    verify_parameter_partition(model, optimizer)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        last_epoch, saved_global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch = last_epoch + 1
        global_step = saved_global_step + 1

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress:
            gt_hr = batch["gt"].to(device=device, dtype=dtype, non_blocking=True)
            oe_hr = batch["oe"].to(device=device, dtype=dtype, non_blocking=True)
            ue_hr = batch["ue"].to(device=device, dtype=dtype, non_blocking=True)
            height, width = gt_hr.shape[-2:]

            gt_lr = F.interpolate(
                gt_hr,
                scale_factor=1 / 3,
                mode="bilinear",
                align_corners=False,
            )
            with torch.no_grad():
                vae_input = (gt_lr * 2 - 1).unsqueeze(2)
                gt_latent = fidelity_vae.encode(
                    vae_input, device=device, tiled=False
                ).to(device=device, dtype=dtype)
                aligned_ue, occ_mask = aligner(
                    ue=ue_hr, target=oe_hr, scale_factor=3
                )
                aligned_ue = aligned_ue * (1 - occ_mask)

            optimizer.zero_grad(set_to_none=True)
            output_hr, output_lr, base_image = model(
                gt_latent=gt_latent,
                oe_hr=oe_hr,
                ue_hr=aligned_ue,
                occ_mask=occ_mask,
                gain_hf=args.gain_hf,
            )

            mse_loss = F.mse_loss(output_hr, gt_hr)
            lpips_loss = (
                perceptual_loss(output_hr, gt_hr, normalize=True).mean() * 0.2
            )
            lap_loss = laplacian_loss(output_hr, gt_hr) * 0.5

            lr_mse_loss = F.mse_loss(output_lr, gt_lr)
            lr_lpips_loss = (
                perceptual_loss(output_lr, gt_lr, normalize=True).mean() * 0.2
            )
            lr_lap_loss = laplacian_loss(output_lr, gt_lr) * 0.5
            loss = (
                mse_loss
                + lpips_loss
                + lap_loss
                + lr_mse_loss
                + lr_lpips_loss
                + lr_lap_loss
            )

            loss.backward()
            verify_frozen_vae_gradients(model)
            optimizer.step()
            scheduler.step()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                size=f"{width}x{height}",
                hr_mse=f"{mse_loss.item():.4f}",
                hr_lpips=f"{lpips_loss.item():.4f}",
                hr_lap=f"{lap_loss.item():.4f}",
                lr_mse=f"{lr_mse_loss.item():.4f}",
                lr_lpips=f"{lr_lpips_loss.item():.4f}",
                lr_lap=f"{lr_lap_loss.item():.4f}",
            )

            if global_step % args.save_every == 0:
                save_image(
                    output_hr.detach().float().clamp(0, 1),
                    output_dir / f"{global_step}_output.jpg",
                )
                save_image(
                    output_lr.detach().float().clamp(0, 1),
                    output_dir / f"{global_step}_output_lr.jpg",
                )
                save_image(
                    base_image.detach().float().clamp(0, 1),
                    output_dir / f"{global_step}_base.jpg",
                )
                save_image(
                    gt_hr.detach().float().clamp(0, 1),
                    output_dir / f"{global_step}_gt.jpg",
                )
                save_checkpoint(
                    output_dir / f"checkpoint-{global_step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    args,
                )
            global_step += 1


if __name__ == "__main__":
    main()
