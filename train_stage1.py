import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from models.utils import load_state_dict
from models.swiftfusion_checkpoints import model_configs_from_directory, tokenizer_config_from_directory
from pipelines.wan_video_new import WanVideoPipeline
from trainers.unified_dataset import UnifiedDataset
from trainers.stage1_utils import DiffusionTrainingModule, ModelLogger, launch_training_task

DEFAULT_LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SwiftFusion exposure-fusion LoRA on Wan 2.1 1.3B.")
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--raft_checkpoint",
        required=True,
        help="RAFT Sintel checkpoint used for global homography alignment.",
    )
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--output_dir", default="./outputs/stage1")
    parser.add_argument("--max_pixels", type=int, default=480 * 640)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument("--gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    return parser.parse_args()


class SwiftFusionTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_dir,
        raft_checkpoint,
        tokenizer_dir=None,
        lora_target_modules=DEFAULT_LORA_TARGETS,
        lora_rank=32,
        lora_checkpoint=None,
        gradient_checkpointing_offload=False,
    ):
        super().__init__()
        tokenizer_config = tokenizer_config_from_directory(model_dir, tokenizer_dir)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs_from_directory(model_dir),
            tokenizer_config=tokenizer_config,
            raft_checkpoint=raft_checkpoint,
        )
        self.configure_swiftfusion_lora(self.pipe, target_modules=lora_target_modules, rank=lora_rank)
        if lora_checkpoint is not None:
            state_dict = load_state_dict(lora_checkpoint)
            incompatible = self.pipe.dit.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"Checkpoint missing trainable parameters: {incompatible.missing_keys}")
        self.use_gradient_checkpointing = True
        self.use_gradient_checkpointing_offload = gradient_checkpointing_offload

    def forward_preprocess(self, data):
        input_video = data["video"]
        if len(input_video) != 2:
            raise ValueError("Each metadata item must provide video=[overexposed, underexposed].")
        inputs_positive = {"prompt": data["prompt"]}
        inputs_negative = {}
        inputs_shared = {
            "gt": data["gt"], "input_video": input_video,
            "height": input_video[0].height, "width": input_video[0].width,
            "num_frames": len(input_video), "cfg_scale": 1, "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False, "vace_scale": 1,
            "max_timestep_boundary": 1.0, "min_timestep_boundary": 0.0,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_positive, inputs_negative = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_positive, inputs_negative)
        return {**inputs_shared, **inputs_positive}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)


def main():
    args = parse_args()
    if not Path(args.metadata_path).is_file():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    if not Path(args.raft_checkpoint).is_file():
        raise FileNotFoundError(
            f"RAFT checkpoint not found: {args.raft_checkpoint}"
        )
    dataset = UnifiedDataset(base_path=args.data_root, metadata_path=args.metadata_path, repeat=args.dataset_repeat, data_file_keys=("gt", "video"), main_data_operator=UnifiedDataset.default_image_operator(base_path=args.data_root, max_pixels=args.max_pixels, height_division_factor=16, width_division_factor=16))
    model = SwiftFusionTrainingModule(
        model_dir=args.model_dir,
        raft_checkpoint=args.raft_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        gradient_checkpointing_offload=(
            args.gradient_checkpointing_offload
        ),
    )
    logger = ModelLogger(args.output_dir, remove_prefix_in_ckpt="pipe.dit.")
    launch_training_task(dataset, model, logger, batch_size=1, learning_rate=args.learning_rate, weight_decay=args.weight_decay, num_workers=args.num_workers, save_steps=args.save_steps, num_epochs=args.epochs, gradient_accumulation_steps=args.gradient_accumulation_steps, find_unused_parameters=args.find_unused_parameters)


if __name__ == "__main__":
    main()
