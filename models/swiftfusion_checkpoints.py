from pathlib import Path

from utils import ModelConfig


WAN21_COMPONENT_FILES = {
    "dit": "diffusion_pytorch_model.safetensors",
    "text_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
    "vae": "Wan2.1_VAE.pth",
}


def model_configs_from_directory(
    model_dir,
    offload_device=None,
    components=("dit", "text_encoder", "vae"),
):
    unknown = sorted(set(components) - set(WAN21_COMPONENT_FILES))
    if unknown:
        raise ValueError(f"Unknown Wan model components: {unknown}")

    model_dir = Path(model_dir)
    paths = [
        model_dir / WAN21_COMPONENT_FILES[component]
        for component in components
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Wan checkpoint files:\n  " + "\n  ".join(missing)
        )
    return [
        ModelConfig(
            path=str(path),
            offload_device=offload_device,
        )
        for path in paths
    ]


def tokenizer_config_from_directory(model_dir, tokenizer_dir=None):
    if tokenizer_dir is not None:
        tokenizer_path = Path(tokenizer_dir)
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(
                f"Tokenizer directory not found: {tokenizer_path}"
            )
        return ModelConfig(path=str(tokenizer_path))

    model_dir = Path(model_dir)
    for candidate in (
        model_dir / "google" / "umt5-xxl",
        model_dir / "google",
    ):
        if candidate.is_dir() and any(candidate.glob("tokenizer*")):
            return ModelConfig(path=str(candidate))
    return None
