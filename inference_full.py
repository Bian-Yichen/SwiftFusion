import argparse
from pathlib import Path

import torch
from PIL import Image

from decoder_utils import FlowAligner as DecoderFlowAligner
from models.decoder_sr import HFBranch, LightWeightPyramidLaplacianSR
from models.trained_decoder import HFConditionedWanVAE, build_arbitrary_sr_cache, pil_pair_to_tensor, run_arbitrary_sr, resize_center_crop
from models.swiftfusion_checkpoints import model_configs_from_directory, tokenizer_config_from_directory
from pipelines.wan_video_new import WanVideoPipeline
from prompts import DEFAULT_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Full four-stage SwiftFusion inference with the trained HF-guided decoder.")
    parser.add_argument("--oe", required=True)
    parser.add_argument("--ue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--raft_checkpoint", required=True)
    parser.add_argument("--base_lora_checkpoint", required=True)
    parser.add_argument("--distilled_lora_checkpoint", required=True)
    parser.add_argument("--sparse_lora_checkpoint", required=True)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dit_height", type=int, default=720)
    parser.add_argument("--dit_width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vram_management", action="store_true")
    return parser.parse_args()


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def save_tensor_image(tensor, path):
    tensor = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu()
    image = Image.fromarray((tensor.numpy() * 255).round().astype("uint8"))
    image.save(path)


def main():
    args = parse_args()
    for path, label in ((args.oe, "OE image"), (args.ue, "UE image"), (args.raft_checkpoint, "RAFT checkpoint"), (args.base_lora_checkpoint, "Stage 1 checkpoint"), (args.distilled_lora_checkpoint, "Stage 3 checkpoint"), (args.sparse_lora_checkpoint, "Stage 4 checkpoint"), (args.decoder_checkpoint, "Decoder checkpoint")):
        require_file(path, label)
    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    oe_image = Image.open(args.oe).convert("RGB")
    ue_image = Image.open(args.ue).convert("RGB")
    if oe_image.size != ue_image.size:
        raise ValueError("OE and UE must have the same input resolution.")
    target_hw = (oe_image.height, oe_image.width)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=model_configs_from_directory(args.model_dir, offload_device="cpu"),
        tokenizer_config=tokenizer_config_from_directory(args.model_dir, args.tokenizer_dir),
        raft_checkpoint=args.raft_checkpoint,
    )
    pipe.dit.enable_block_sparse_attention(layer_stride=2)
    pipe.load_lora(pipe.dit, args.base_lora_checkpoint, alpha=1.0)
    pipe.load_lora(pipe.dit, args.distilled_lora_checkpoint, alpha=1.0, remove_prefix="pipe.dit.", delete_prefix="fake_teacher.dit.")
    pipe.load_lora(pipe.dit, args.sparse_lora_checkpoint, alpha=1.0)

    conditioned_vae = HFConditionedWanVAE(pipe.vae, args.decoder_checkpoint)
    pipe.vae = conditioned_vae
    decoder_flow_aligner = DecoderFlowAligner(args.raft_checkpoint).to("cuda")

    dit_pair = [resize_center_crop(oe_image, args.dit_height, args.dit_width), resize_center_crop(ue_image, args.dit_height, args.dit_width)]
    vae_condition = pil_pair_to_tensor([oe_image, ue_image], args.dit_height * 3, args.dit_width * 3, "cuda", torch.float32)
    aligned_ue, occ_mask = decoder_flow_aligner(ue=vae_condition[1:2], target=vae_condition[0:1], scale_factor=3)
    conditioned_vae.set_condition(vae_condition[0:1].to(torch.bfloat16), aligned_ue.to(torch.bfloat16), occ_mask.to(torch.bfloat16))

    pipe.eval()
    if args.vram_management:
        pipe.enable_vram_management()
    with torch.inference_mode():
        video, _, _ = pipe(prompt=args.prompt, negative_prompt="", input_video=dit_pair, height=args.dit_height, width=args.dit_width, num_frames=2, cfg_scale=1, seed=args.seed, num_inference_steps=1, progress_bar_cmd=lambda timesteps: timesteps)

    decoder_flow_aligner.cpu()
    if args.vram_management:
        pipe.load_models_to_device([])
    del decoder_flow_aligner, conditioned_vae, pipe
    torch.cuda.empty_cache()

    decoder_checkpoint = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
    decoder_state = decoder_checkpoint.get("trainable_model", decoder_checkpoint)
    hf_branch = HFBranch(sr_channels=256, hf_channels=64).cuda().to(dtype=torch.bfloat16)
    hf_branch.load_state_dict(decoder_state["hf_branch"], strict=True)
    hf_branch.eval().requires_grad_(False)
    sr_model = LightWeightPyramidLaplacianSR(feat_channels=256).cuda().to(dtype=torch.bfloat16)
    sr_model.load_state_dict(decoder_state["light_weight_pyramid_laplacian_sr"], strict=True)
    sr_model.eval().requires_grad_(False)

    target_inputs = pil_pair_to_tensor([oe_image, ue_image], target_hw[0], target_hw[1], "cuda", torch.bfloat16)
    target_aligner = DecoderFlowAligner(args.raft_checkpoint).cuda()
    aligned_target_ue, target_occ = target_aligner(ue=target_inputs[1:2], target=target_inputs[0:1], scale_factor=3)
    target_aligner.cpu()
    del target_aligner
    torch.cuda.empty_cache()

    lr_image = ((video[:, :, 0] + 1.0) * 0.5).to(dtype=torch.bfloat16)
    with torch.inference_mode():
        sr_cache = build_arbitrary_sr_cache(hf_branch, target_inputs[0:1], aligned_target_ue, target_occ, target_hw)
        output_hr, _ = run_arbitrary_sr(sr_model, lr_image, sr_cache, target_hw)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor_image(lr_image, output_path.with_name(output_path.stem + "_720p.png"))
    save_tensor_image(output_hr, output_path)
    oe_image.save(output_path.with_name(output_path.stem + "_oe.png"))
    ue_image.save(output_path.with_name(output_path.stem + "_ue.png"))


if __name__ == "__main__":
    main()
