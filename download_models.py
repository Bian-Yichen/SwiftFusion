import argparse

from modelscope import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the Wan 2.1 1.3B files required by SwiftFusion."
    )
    parser.add_argument(
        "--output_dir",
        default="./models/Wan-AI/Wan2.1-T2V-1.3B",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_download(
        "Wan-AI/Wan2.1-T2V-1.3B",
        local_dir=args.output_dir,
        allow_file_pattern=[
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/*",
        ],
    )


if __name__ == "__main__":
    main()
