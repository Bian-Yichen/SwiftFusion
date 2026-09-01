import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm


class DiffusionTrainingModule(torch.nn.Module):
    def to(self, *args, **kwargs):
        for model in self.children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_param_names(self):
        return {name for name, parameter in self.named_parameters() if parameter.requires_grad}

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.to(upcast_dtype)
        return model

    def configure_swiftfusion_lora(self, pipe, target_modules, rank):
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except([])
        pipe.dit = self.add_lora_to_model(pipe.dit, target_modules=target_modules.split(","), lora_rank=rank, upcast_dtype=pipe.torch_dtype)
        pipe.dit.hdr_cond_patch_embedding.requires_grad_(True)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = self.trainable_param_names()
        state_dict = {name: parameter for name, parameter in state_dict.items() if name in trainable_names}
        if remove_prefix is None:
            return state_dict
        return {(name[len(remove_prefix):] if name.startswith(remove_prefix) else name): parameter for name, parameter in state_dict.items()}


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda state_dict: state_dict):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def _state_dict(self, accelerator, model):
        state_dict = accelerator.get_state_dict(model)
        state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
        return self.state_dict_converter(state_dict)

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        os.makedirs(self.output_path, exist_ok=True)
        accelerator.save(self._state_dict(accelerator, model), os.path.join(self.output_path, file_name), safe_serialization=True)

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


def launch_training_task(dataset, model, model_logger, batch_size=1, learning_rate=1e-4, weight_decay=1e-2, num_workers=0, save_steps=None, num_epochs=1, gradient_accumulation_steps=1, find_unused_parameters=False):
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch[0], num_workers=num_workers)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)])
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    for epoch_id in range(num_epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch_id}", disable=not accelerator.is_local_main_process)
        for data in progress:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                if accelerator.is_main_process:
                    progress.set_postfix(loss=f"{loss.item():.4f}")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
