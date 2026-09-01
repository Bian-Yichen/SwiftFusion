from typing import Any

import torch


def tokenize_long_prompt(tokenizer, prompt, max_length=None):
    length = tokenizer.model_max_length if max_length is None else max_length
    tokenizer.model_max_length = 99_999_999
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    padded_length = (input_ids.shape[1] + length - 1) // length * length
    tokenizer.model_max_length = length
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=padded_length,
        truncation=True,
    ).input_ids
    return input_ids.reshape((input_ids.shape[1] // length, length))


class BasePrompter:
    def __init__(self):
        self.refiners = []
        self.extenders = []

    def load_prompt_refiners(self, model_pool: Any, refiner_classes=()):
        for refiner_class in refiner_classes:
            self.refiners.append(refiner_class.from_model_manager(model_pool))

    def load_prompt_extenders(self, model_pool: Any, extender_classes=()):
        for extender_class in extender_classes:
            self.extenders.append(extender_class.from_model_manager(model_pool))

    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            return [
                self.process_prompt(item, positive=positive)
                for item in prompt
            ]
        for refiner in self.refiners:
            prompt = refiner(prompt, positive=positive)
        return prompt

    @torch.no_grad()
    def extend_prompt(self, prompt: str, positive=True):
        extended_prompt = {"prompt": prompt}
        for extender in self.extenders:
            extended_prompt = extender(extended_prompt)
        return extended_prompt
