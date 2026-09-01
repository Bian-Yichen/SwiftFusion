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
from trainers.distill_utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_distillation_task,
)


DEFAULT_LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SwiftFusion with one-step distribution matching."
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
        "--decoder_checkpoint",
        required=True,
        help="HF-guided decoder checkpoint trained by stage 2.",
    )
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="./outputs/stage3")
    parser.add_argument("--max_pixels", type=int, default=480 * 640)
    parser.add_argument("--decoder_max_pixels", type=int, default=3840 * 2160)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--fake_teacher_learning_rate",
        type=float,
        default=None,
    )
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
        help="Optional Stage 3 checkpoint containing student and fake teacher.",
    )
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument(
        "--gradient_checkpointing_offload",
        action="store_true",
    )
    return parser.parse_args()


class SwiftFusionDistillationModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_dir,
        raft_checkpoint,
        base_lora_checkpoint,
        decoder_checkpoint,
        tokenizer_dir=None,
        lora_target_modules=DEFAULT_LORA_TARGETS,
        lora_rank=32,
        resume_checkpoint=None,
        min_timestep_boundary=0.0,
        max_timestep_boundary=1.0,
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
        self.real_teacher = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs_from_directory(
                model_dir,
                components=("dit",),
            ),
            tokenizer_config=None,
            required_models=("dit",),
        )
        self.fake_teacher = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs_from_directory(
                model_dir,
                components=("dit",),
            ),
            tokenizer_config=None,
            required_models=("dit",),
        )

        for pipeline in (
            self.pipe,
            self.real_teacher,
            self.fake_teacher,
        ):
            pipeline.load_lora(
                pipeline.dit,
                base_lora_checkpoint,
                alpha=1.0,
            )

        self.real_teacher.freeze_except([])
        self.real_teacher.eval()
        self.configure_swiftfusion_lora(
            self.pipe,
            lora_target_modules,
            lora_rank,
        )
        self.configure_swiftfusion_lora(
            self.fake_teacher,
            lora_target_modules,
            lora_rank,
        )

        if resume_checkpoint is not None:
            self.pipe.load_lora(
                self.pipe.dit,
                resume_checkpoint,
                alpha=1.0,
                remove_prefix="pipe.dit.",
                delete_prefix="fake_teacher.dit.",
            )
            self.fake_teacher.load_lora(
                self.fake_teacher.dit,
                resume_checkpoint,
                alpha=1.0,
                remove_prefix="fake_teacher.dit.",
                delete_prefix="pipe.dit.",
            )

        self.net_lpips = lpips.LPIPS(net="vgg")
        self.net_lpips.requires_grad_(False)
        self.net_lpips.eval()

        self.min_timestep_boundary = min_timestep_boundary
        self.max_timestep_boundary = max_timestep_boundary
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

    def student_forward(self, data):
        input_video = data["video"]
        if len(input_video) != 2:
            raise ValueError(
                "Each item must contain video=[overexposed, underexposed]."
            )
        self.set_decoder_condition(data, input_video)
        video, inputs_shared, inputs_positive = self.pipe(
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
            return_dmd_inputs=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=(
                self.gradient_checkpointing_offload
            ),
        )

        clean_latents = inputs_shared["latents"]
        device = clean_latents.device
        dtype = clean_latents.dtype
        min_step = int(
            self.min_timestep_boundary
            * self.fake_teacher.scheduler.num_train_timesteps
        )
        max_step = int(
            self.max_timestep_boundary
            * self.fake_teacher.scheduler.num_train_timesteps
        )
        timestep_id = torch.randint(
            min_step,
            max_step,
            (1,),
            device=self.fake_teacher.scheduler.timesteps.device,
        )
        timestep = self.fake_teacher.scheduler.timesteps[
            timestep_id
        ].to(device=device, dtype=dtype)
        noise = self.pipe.generate_noise(
            clean_latents.shape,
            rand_torch_dtype=dtype,
            rand_device=device,
            torch_dtype=dtype,
            device=device,
        )
        noised_latents = self.fake_teacher.scheduler.add_noise(
            clean_latents,
            noise,
            timestep,
        )
        teacher_inputs = {
            **inputs_shared,
            "input_latents": clean_latents,
            "latents": noised_latents,
            "noise": noise,
        }
        score_inputs = {
            **teacher_inputs,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
        }
        with torch.no_grad():
            real_velocity = self.real_teacher.model_fn(
                dit=self.real_teacher.dit,
                **score_inputs,
                **inputs_positive,
                timestep=timestep,
            )
            fake_velocity = self.fake_teacher.model_fn(
                dit=self.fake_teacher.dit,
                **score_inputs,
                **inputs_positive,
                timestep=timestep,
            )
        score_gradient = fake_velocity - real_velocity

        dmd_loss = F.mse_loss(
            noised_latents,
            (noised_latents - score_gradient).detach(),
        )
        target = inputs_shared["gt_tensor"]
        l2_loss = F.mse_loss(video.float(), target.float())
        lpips_loss = (
            self.net_lpips(
                video.squeeze(2).float(),
                target.squeeze(2).float(),
            ).mean()
            * 0.5
        )
        return {
            "dmd": dmd_loss,
            "l2": l2_loss,
            "lpips": lpips_loss,
            "teacher_inputs": teacher_inputs,
            "inputs_positive": inputs_positive,
            "timestep": timestep,
        }

    def fake_teacher_forward(
        self,
        teacher_inputs,
        inputs_positive,
        timestep,
    ):
        inputs = dict(teacher_inputs)
        clean_latents = inputs["input_latents"]
        noise = self.fake_teacher.generate_noise(
            clean_latents.shape,
            rand_torch_dtype=clean_latents.dtype,
            rand_device=clean_latents.device,
            torch_dtype=clean_latents.dtype,
            device=clean_latents.device,
        )
        inputs["noise"] = noise
        inputs["latents"] = self.fake_teacher.scheduler.add_noise(
            clean_latents,
            noise,
            timestep,
        )
        inputs["use_gradient_checkpointing"] = True
        inputs["use_gradient_checkpointing_offload"] = (
            self.gradient_checkpointing_offload
        )
        target = self.fake_teacher.scheduler.training_target(
            clean_latents,
            noise,
        )
        prediction = self.fake_teacher.model_fn(
            dit=self.fake_teacher.dit,
            **inputs,
            **inputs_positive,
            timestep=timestep,
        )
        loss = F.mse_loss(
            prediction.float(),
            target.float(),
        )
        return loss * self.fake_teacher.scheduler.training_weight(
            timestep
        )

    def forward(
        self,
        data,
        mode="student",
        teacher_inputs=None,
        inputs_positive=None,
        timestep=None,
    ):
        if mode == "student":
            return self.student_forward(data)
        if mode == "fake_teacher":
            return self.fake_teacher_forward(
                teacher_inputs,
                inputs_positive,
                timestep,
            )
        raise ValueError(f"Unknown training mode: {mode}")


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.metadata_path, "Metadata")
    require_file(args.raft_checkpoint, "RAFT checkpoint")
    require_file(args.base_lora_checkpoint, "v1.6 checkpoint")
    require_file(args.decoder_checkpoint, "Decoder checkpoint")
    if args.resume_checkpoint is not None:
        require_file(args.resume_checkpoint, "Resume checkpoint")
    if not (
        0
        <= args.min_timestep_boundary
        < args.max_timestep_boundary
        <= 1
    ):
        raise ValueError(
            "Timestep boundaries must satisfy 0 <= min < max <= 1."
        )

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
    dataset = DecoderConditionedDataset(
        low_dataset,
        high_dataset,
    )
    model = SwiftFusionDistillationModule(
        model_dir=args.model_dir,
        raft_checkpoint=args.raft_checkpoint,
        base_lora_checkpoint=args.base_lora_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        resume_checkpoint=args.resume_checkpoint,
        min_timestep_boundary=args.min_timestep_boundary,
        max_timestep_boundary=args.max_timestep_boundary,
        gradient_checkpointing_offload=(
            args.gradient_checkpointing_offload
        ),
    )
    logger = ModelLogger(args.output_dir)
    launch_distillation_task(
        dataset,
        model,
        logger,
        learning_rate=args.learning_rate,
        fake_teacher_learning_rate=(
            args.fake_teacher_learning_rate
        ),
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        save_steps=args.save_steps,
        num_epochs=args.epochs,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
    )


if __name__ == "__main__":
    main()
