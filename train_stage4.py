import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lpips
import torch
import torch.nn.functional as F

from decoder_utils import FlowAligner as DecoderFlowAligner
from models.trained_decoder import (
    DecoderConditionedDataset,
    HFConditionedWanVAE,
    pil_pair_to_tensor,
)
from models.swiftfusion_checkpoints import (
    model_configs_from_directory,
    tokenizer_config_from_directory,
)
from pipelines.wan_video_new import WanVideoPipeline
from prompts import DEFAULT_PROMPT
from trainers.unified_dataset import UnifiedDataset
from trainers.utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_training_task,
)


DEFAULT_LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SwiftFusion with block-sparse self-attention."
    )
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--raft_checkpoint", required=True)
    parser.add_argument(
        "--base_lora_checkpoint",
        required=True,
        help="LoRA checkpoint produced by Stage 1.",
    )
    parser.add_argument(
        "--distilled_lora_checkpoint",
        required=True,
        help="Student checkpoint produced by Stage 3.",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        required=True,
        help="HF-guided decoder checkpoint trained by stage 2.",
    )
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="./outputs/stage4")
    parser.add_argument("--max_pixels", type=int, default=512 * 640)
    parser.add_argument("--decoder_max_pixels", type=int, default=3840 * 2160)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        default=DEFAULT_LORA_TARGETS,
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Optional LoRA checkpoint produced by Stage 4.",
    )
    parser.add_argument(
        "--gradient_checkpointing_offload",
        action="store_true",
    )
    parser.add_argument("--find_unused_parameters", action="store_true")
    return parser.parse_args()


class SwiftFusionSparseTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_dir,
        raft_checkpoint,
        base_lora_checkpoint,
        distilled_lora_checkpoint,
        decoder_checkpoint,
        tokenizer_dir=None,
        lora_target_modules=DEFAULT_LORA_TARGETS,
        lora_rank=32,
        resume_checkpoint=None,
        gradient_checkpointing_offload=False,
    ):
        super().__init__()
        tokenizer_config = tokenizer_config_from_directory(
            model_dir,
            tokenizer_dir,
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs_from_directory(model_dir),
            tokenizer_config=tokenizer_config,
            raft_checkpoint=raft_checkpoint,
        )
        self.pipe.vae = HFConditionedWanVAE(
            self.pipe.vae,
            decoder_checkpoint,
        )
        self.decoder_flow_aligner = DecoderFlowAligner(raft_checkpoint)
        self.pipe.dit.enable_block_sparse_attention(
            layer_stride=2,
        )
        self.pipe.load_lora(
            self.pipe.dit,
            base_lora_checkpoint,
            alpha=1.0,
        )
        self.pipe.load_lora(
            self.pipe.dit,
            distilled_lora_checkpoint,
            alpha=1.0,
            remove_prefix="pipe.dit.",
            delete_prefix="fake_teacher.dit.",
        )
        self.configure_swiftfusion_lora(
            self.pipe,
            lora_target_modules,
            lora_rank,
        )
        if resume_checkpoint is not None:
            self.pipe.load_lora(
                self.pipe.dit,
                resume_checkpoint,
                alpha=1.0,
            )

        self.net_lpips = lpips.LPIPS(net="vgg")
        self.net_lpips.requires_grad_(False)
        self.net_lpips.eval()
        self.gradient_checkpointing_offload = (
            gradient_checkpointing_offload
        )

    @torch.no_grad()
    def set_decoder_condition(self, data, input_video):
        target_height = input_video[0].height * 3
        target_width = input_video[0].width * 3
        device = next(self.pipe.dit.parameters()).device
        high = pil_pair_to_tensor(
            data["decoder_video"],
            target_height,
            target_width,
            device,
            torch.float32,
        )
        oe, ue = high[0:1], high[1:2]
        self.decoder_flow_aligner.to(device)
        aligned_ue, occ_mask = self.decoder_flow_aligner(
            ue=ue,
            target=oe,
            scale_factor=3,
        )
        self.pipe.vae.set_condition(
            oe.to(self.pipe.torch_dtype),
            aligned_ue.to(self.pipe.torch_dtype),
            occ_mask.to(self.pipe.torch_dtype),
        )

    def forward(self, data):
        input_video = data["video"]
        if len(input_video) != 2:
            raise ValueError(
                "Each item must contain video=[overexposed, underexposed]."
            )
        self.set_decoder_condition(data, input_video)
        video, noise_prediction, inputs = self.pipe(
            prompt=data.get("prompt", DEFAULT_PROMPT),
            negative_prompt="",
            input_video=input_video,
            gt=data["gt"],
            height=input_video[0].height,
            width=input_video[0].width,
            num_frames=2,
            cfg_scale=1,
            num_inference_steps=1,
            progress_bar_cmd=lambda timesteps: timesteps,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=(
                self.gradient_checkpointing_offload
            ),
        )
        target = self.pipe.scheduler.training_target(
            inputs["input_latents"],
            inputs["noise"]
        )
        gt_tensor = inputs["gt_tensor"]
        return {
            "flow": F.mse_loss(
                noise_prediction.float(),
                target.float(),
            ),
            "l2": F.mse_loss(
                video.float(),
                gt_tensor.float(),
            ),
            "lpips": (
                self.net_lpips(
                    video.squeeze(2).float(),
                    gt_tensor.squeeze(2).float(),
                ).mean()
                * 0.5
            ),
        }


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.metadata_path, "Metadata")
    require_file(args.raft_checkpoint, "RAFT checkpoint")
    require_file(args.base_lora_checkpoint, "v1.6 checkpoint")
    require_file(
        args.distilled_lora_checkpoint,
        "v1.19 checkpoint",
    )
    require_file(args.decoder_checkpoint, "Decoder checkpoint")
    if args.resume_checkpoint is not None:
        require_file(args.resume_checkpoint, "Resume checkpoint")

    low_dataset = UnifiedDataset(
        base_path=args.data_root,
        metadata_path=args.metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=("gt", "video"),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.data_root,
            max_pixels=args.max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    high_dataset = UnifiedDataset(
        base_path=args.data_root,
        metadata_path=args.metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=("video",),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.data_root,
            max_pixels=args.decoder_max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    dataset = DecoderConditionedDataset(low_dataset, high_dataset)
    model = SwiftFusionSparseTrainingModule(
        model_dir=args.model_dir,
        raft_checkpoint=args.raft_checkpoint,
        base_lora_checkpoint=args.base_lora_checkpoint,
        distilled_lora_checkpoint=args.distilled_lora_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.resume_checkpoint,
        gradient_checkpointing_offload=(
            args.gradient_checkpointing_offload
        ),
    )
    logger = ModelLogger(
        args.output_dir,
        remove_prefix_in_ckpt="pipe.dit.",
    )
    launch_training_task(
        dataset,
        model,
        logger,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        save_steps=args.save_steps,
        num_epochs=args.epochs,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        find_unused_parameters=args.find_unused_parameters,
    )


if __name__ == "__main__":
    main()
