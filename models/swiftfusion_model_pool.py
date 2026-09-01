from diffsynth.models.model_loader import ModelPool


SWIFTFUSION_MODEL_CLASSES = {
    # Wan 2.1 / 2.2 DiT checkpoints supported by DiffSynth 2.1.1.
    "9269f8db9040a9d860eaca435be61814": "models.wan_video_modules.wan_video_dit.WanModel",
    "aafcfd9672c3a2456dc46e1cb6e52c70": "models.wan_video_modules.wan_video_dit.WanModel",
    "6bfcfb3b342cb286ce886889d519a77e": "models.wan_video_modules.wan_video_dit.WanModel",
    "6d6ccde6845b95ad9114ab993d917893": "models.wan_video_modules.wan_video_dit.WanModel",
    "349723183fc063b2bfc10bb2835cf677": "models.wan_video_modules.wan_video_dit.WanModel",
    "efa44cddf936c70abd0ea28b6cbe946c": "models.wan_video_modules.wan_video_dit.WanModel",
    "3ef3b1f8e1dab83d5b71fd7b617f859f": "models.wan_video_modules.wan_video_dit.WanModel",
    "70ddad9d3a133785da5ea371aae09504": "models.wan_video_modules.wan_video_dit.WanModel",
    "26bde73488a92e64cc20b0a7485b9e5b": "models.wan_video_modules.wan_video_dit.WanModel",
    "ac6a5aa74f4a0aab6f64eb9a72f19901": "models.wan_video_modules.wan_video_dit.WanModel",
    "b61c605c2adbd23124d152ed28e049ae": "models.wan_video_modules.wan_video_dit.WanModel",
    "1f5ab7703c6fc803fdded85ff040c316": "models.wan_video_modules.wan_video_dit.WanModel",
    "5b013604280dd715f8457c6ed6d6a626": "models.wan_video_modules.wan_video_dit.WanModel",
    "2267d489f0ceb9f21836532952852ee5": "models.wan_video_modules.wan_video_dit.WanModel",
    "5ec04e02b42d2580483ad69f4e76346a": "models.wan_video_modules.wan_video_dit.WanModel",
    "47dbeab5e560db3180adf51dc0232fb1": "models.wan_video_modules.wan_video_dit.WanModel",
    "9c8818c2cbea55eca56c7b447df170da": "models.wan_video_modules.wan_video_text_encoder.WanTextEncoder",
    "1378ea763357eea97acdef78e65d6d96": "models.wan_video_modules.wan_video_vae.WanVideoVAE",
    "ccc42284ea13e1ad04693284c7a09be6": "models.wan_video_modules.wan_video_vae.WanVideoVAE",
    "e1de6c02cdac79f8b739f4d3698cd216": "models.wan_video_modules.wan_video_vae.WanVideoVAE38",
}


class SwiftFusionModelPool(ModelPool):
    """DiffSynth 2.1.1 model pool backed by SwiftFusion's conditioned models."""

    def load_model_file(
        self,
        config,
        path,
        vram_config,
        vram_limit=None,
        state_dict=None,
        quantize=None,
    ):
        local_model_class = SWIFTFUSION_MODEL_CLASSES.get(config.get("model_hash"))
        if local_model_class is not None:
            config = {**config, "model_class": local_model_class}
        return super().load_model_file(
            config,
            path,
            vram_config,
            vram_limit=vram_limit,
            state_dict=state_dict,
            quantize=quantize,
        )


def load_swiftfusion_model_configs(model_configs, torch_dtype, device, use_usp=False):
    """Download configured files if needed and load them through ModelPool."""
    model_pool = SwiftFusionModelPool()
    for model_config in model_configs:
        model_config.download_if_necessary(use_usp=use_usp)
        load_device = model_config.offload_device or device
        load_dtype = model_config.offload_dtype or torch_dtype
        vram_config = model_pool.default_vram_config()
        vram_config.update(
            {
                "onload_dtype": load_dtype,
                "onload_device": load_device,
                "preparing_dtype": load_dtype,
                "preparing_device": load_device,
                "computation_dtype": load_dtype,
                "computation_device": load_device,
            }
        )
        model_pool.auto_load_model(
            model_config.path,
            vram_config=vram_config,
        )
    return model_pool
