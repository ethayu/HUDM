from __future__ import annotations

import re
from typing import Any


def remap_hf_vit_encoder_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate HF ViT encoder keys to the custom ViT key layout."""
    hf_attention_map = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
    }
    hf_layer_re = re.compile(r"^encoder\.encoder\.layer\.(\d+)\.(.*)")

    def remap_key(key: str) -> str:
        match = hf_layer_re.match(key)
        if not match:
            return key
        idx, rest = match.group(1), match.group(2)
        for hf_prefix, custom_prefix in hf_attention_map.items():
            if rest == hf_prefix or rest.startswith(f"{hf_prefix}."):
                rest = custom_prefix + rest[len(hf_prefix):]
                break
        return f"encoder.layers.{idx}.{rest}"

    if not any(hf_layer_re.match(key) for key in state_dict):
        return state_dict
    return {remap_key(key): value for key, value in state_dict.items()}


def remap_custom_vit_encoder_keys_to_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate custom Le-WM ViT encoder keys to the HF ViT key layout."""
    custom_attention_map = {
        "attention.q_proj": "attention.attention.query",
        "attention.k_proj": "attention.attention.key",
        "attention.v_proj": "attention.attention.value",
        "attention.o_proj": "attention.output.dense",
        "mlp.fc1": "intermediate.dense",
        "mlp.fc2": "output.dense",
    }
    custom_layer_re = re.compile(r"^encoder\.layers\.(\d+)\.(.*)")

    def remap_key(key: str) -> str:
        match = custom_layer_re.match(key)
        if not match:
            return key
        idx, rest = match.group(1), match.group(2)
        for custom_prefix, hf_prefix in custom_attention_map.items():
            if rest == custom_prefix or rest.startswith(f"{custom_prefix}."):
                rest = hf_prefix + rest[len(custom_prefix):]
                break
        return f"encoder.encoder.layer.{idx}.{rest}"

    if not any(custom_layer_re.match(key) for key in state_dict):
        return state_dict
    return {remap_key(key): value for key, value in state_dict.items()}


def remap_vit_encoder_keys_for_model(state_dict: dict[str, Any], model: Any) -> dict[str, Any]:
    """Map serialized ViT encoder keys into the instantiated model's key layout."""
    model_keys = set(model.state_dict())
    state_keys = set(state_dict)
    if state_keys <= model_keys:
        return state_dict

    model_uses_hf = any(key.startswith("encoder.encoder.layer.") for key in model_keys)
    model_uses_custom = any(key.startswith("encoder.layers.") for key in model_keys)
    state_uses_hf = any(key.startswith("encoder.encoder.layer.") for key in state_keys)
    state_uses_custom = any(key.startswith("encoder.layers.") for key in state_keys)

    if state_uses_hf and model_uses_custom:
        return remap_hf_vit_encoder_keys(state_dict)
    if state_uses_custom and model_uses_hf:
        return remap_custom_vit_encoder_keys_to_hf(state_dict)
    return state_dict


__all__ = [
    "remap_custom_vit_encoder_keys_to_hf",
    "remap_hf_vit_encoder_keys",
    "remap_vit_encoder_keys_for_model",
]
