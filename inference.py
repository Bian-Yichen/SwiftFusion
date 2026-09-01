import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models.swiftfusion_checkpoints import (
    model_configs_from_directory,
    tokenizer_config_from_directory,
)
from pipelines.wan_video_new import WanVideoPipeline
from prompts import DEFAULT_PROMPT
from trainers.unified_dataset import UnifiedDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run block-sparse SwiftFusion v1.38 inference."
    )
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--raft_checkpoint", required=True)
    parser.add_argument("--base_lora_checkpoint", required=True)
    parser.add_argument("--distilled_lora_checkpoint", required=True)
    parser.add_argument("--sparse_lora_checkpoint", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="./outputs/inference")
    parser.add_argument("--max_pixels", type=int, default=512 * 640)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    args = parse_args()
    checkpoints = (
        (args.metadata_path, "Metadata"),
        (args.raft_checkpoint, "RAFT checkpoint"),
        (args.base_lora_checkpoint, "Stage 1 checkpoint"),
        (
            args.distilled_lora_checkpoint,
            "Stage 3 checkpoint",
        ),
        (args.sparse_lora_checkpoint, "Stage 4 checkpoint"),
    )
    for path, label in checkpoints:
        require_file(path, label)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs_from_directory(
            args.model_dir,
            offload_device="cpu",
        ),
        tokenizer_config=tokenizer_config_from_directory(
            args.model_dir,
            args.tokenizer_dir,
        ),
        raft_checkpoint=args.raft_checkpoint,
    )
    pipe.dit.enable_block_sparse_attention(
        layer_stride=2,
    )
    pipe.load_lora(
        pipe.dit,
        args.base_lora_checkpoint,
        alpha=1.0,
    )
    pipe.load_lora(
        pipe.dit,
        args.distilled_lora_checkpoint,
        alpha=1.0,
        remove_prefix="pipe.dit.",
        delete_prefix="fake_teacher.dit.",
    )
    pipe.load_lora(
        pipe.dit,
        args.sparse_lora_checkpoint,
        alpha=1.0,
    )
    pipe.eval()
    pipe.enable_vram_management()

    dataset = UnifiedDataset(
        base_path=args.data_root,
        metadata_path=args.metadata_path,
        repeat=1,
        data_file_keys=("gt", "video"),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.data_root,
            max_pixels=args.max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=args.num_workers,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for step, data in enumerate(tqdm(dataloader)):
            input_video = data["video"]
            output, _, _ = pipe(
                prompt=data.get("prompt", DEFAULT_PROMPT),
                negative_prompt="",
                input_video=input_video,
                height=input_video[0].height,
                width=input_video[0].width,
                num_frames=2,
                cfg_scale=1,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
            progress_bar_cmd=lambda timesteps: timesteps,
            )
            result = pipe.vae_output_to_video(output)[0]
            input_video[0].save(output_dir / f"{step}_oe.png")
            input_video[1].save(output_dir / f"{step}_ue.png")
            if "gt" in data:
                data["gt"].save(output_dir / f"{step}_gt.png")
            result.save(output_dir / f"{step}_result.png")


if __name__ == "__main__":
    main()
