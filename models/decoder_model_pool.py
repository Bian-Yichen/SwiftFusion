import torch

from diffsynth.models.model_loader import ModelPool


WAN21_VAE_MODEL_HASH = "ccc42284ea13e1ad04693284c7a09be6"
LOCAL_WAN21_VAE_CLASS = (
    "models.wan_video_modules.wan_video_vae.WanVideoVAE"
)


class DecoderModelPool(ModelPool):
    """DiffSynth 2.1.1 model pool using SwiftFusion's conditioned Wan VAE class."""

    def load_model_file(
        self,
        config,
        path,
        vram_config,
        vram_limit=None,
        state_dict=None,
        quantize=None,
    ):
        if config.get("model_hash") == WAN21_VAE_MODEL_HASH:
            config = {**config, "model_class": LOCAL_WAN21_VAE_CLASS}
        return super().load_model_file(
            config,
            path,
            vram_config,
            vram_limit=vram_limit,
            state_dict=state_dict,
            quantize=quantize,
        )


def load_decoder_wan_vae(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    model_pool = DecoderModelPool()
    vram_config = model_pool.default_vram_config()
    vram_config.update(
        {
            "onload_dtype": dtype,
            "onload_device": device,
            "preparing_dtype": dtype,
            "preparing_device": device,
            "computation_dtype": dtype,
            "computation_device": device,
        }
    )
    model_pool.auto_load_model(checkpoint_path, vram_config=vram_config)
    vae = model_pool.fetch_model("wan_video_vae")
    if vae is None:
        raise RuntimeError(f"Failed to load a Wan VAE from {checkpoint_path}")
    if vae.__class__.__module__ != "models.wan_video_modules.wan_video_vae":
        raise RuntimeError(
            "DiffSynth loaded its stock Wan VAE instead of SwiftFusion's "
            "HF-conditioned Wan VAE class."
        )
    return vae
