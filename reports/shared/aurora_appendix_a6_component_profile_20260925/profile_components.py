"""Profile the Aurora MWM image encoder path and latent-prefix decoders.

Instantiates exact module architectures from a saved checkpoint config. Weights
do not affect parameter counts, operator FLOPs, or the shapes of these calls.
No training or checkpoint weights are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from mwm.models.decoders import ConvImageDecoder


LEVELS = (48, 72, 96, 120, 144, 168, 192)


class EncoderPath(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, projector: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        return self.projector(output.last_hidden_state[:, 0])


def count_params(module: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def count_flops(module: torch.nn.Module, inputs: torch.Tensor) -> int:
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        module(inputs)
    return int(counter.get_total_flops())


def time_cuda_events(
    module: torch.nn.Module, inputs: torch.Tensor, *, warmup: int, repeats: int
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(inputs)
        torch.cuda.synchronize()
        pairs = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(inputs)
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
    elapsed = sorted(float(start.elapsed_time(end)) for start, end in pairs)
    return {
        "median_ms": statistics.median(elapsed),
        "min_ms": elapsed[0],
        "max_ms": elapsed[-1],
        "p10_ms": elapsed[int((len(elapsed) - 1) * 0.10)],
        "p90_ms": elapsed[int((len(elapsed) - 1) * 0.90)],
    }


def profile(module: torch.nn.Module, inputs: torch.Tensor, *, warmup: int, repeats: int) -> dict:
    module.eval().to(device="cuda", dtype=torch.float32)
    inputs = inputs.to(device="cuda", dtype=torch.float32)
    result = {
        "input_shape": list(inputs.shape),
        "params": count_params(module),
        "flops_per_pass": count_flops(module, inputs),
        "latency": time_cuda_events(module, inputs, warmup=warmup, repeats=repeats),
    }
    torch.cuda.synchronize()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    assert args.warmup >= 1 and args.repeats >= 5
    assert torch.cuda.is_available(), "CUDA device required for timing"
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    source = config["kwargs"]["source_config"]
    image_shape = tuple(config["kwargs"]["image_shape"])
    assert image_shape == (224, 224), image_shape

    encoder = instantiate(source["encoder"])
    projector = instantiate(source["projector"])
    path = EncoderPath(encoder, projector)
    torch.manual_seed(3072)
    pixels = torch.rand(1, 3, *image_shape)
    results = {
        "protocol": {
            "source_config": str(args.config),
            "checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_config_sha256": config["kwargs"]["source_config_sha256"],
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "precision": "float32",
            "mode": "eval + inference_mode",
            "batch_size": 1,
            "warmup_passes": args.warmup,
            "timed_passes": args.repeats,
            "timing": "CUDA event elapsed time per forward pass; median reported",
            "flops": "torch.utils.flop_counter.FlopCounterMode total; one forward pass",
            "input_preprocessing": "excluded (normalized 3x224x224 tensor supplied)",
            "weights": "architecture instantiated from saved config; weights not loaded",
        },
        "encoder_backbone_params": count_params(encoder),
        "projector_params": count_params(projector),
        "encoder_path": profile(path, pixels, warmup=args.warmup, repeats=args.repeats),
        "decoders": {},
    }
    for level in LEVELS:
        decoder = ConvImageDecoder(latent_dim=level, image_shape=image_shape)
        latent = torch.randn(1, level)
        results["decoders"][str(level)] = profile(
            decoder, latent, warmup=args.warmup, repeats=args.repeats
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
