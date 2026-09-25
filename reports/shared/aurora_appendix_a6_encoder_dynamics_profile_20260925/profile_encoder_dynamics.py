"""Architecture-only encoder and one-step LeWM dynamics component profile.

The seven-level MWM is instantiated from the saved TwoRoom Single-K=192 base
source configuration, using the released MWM level list and adapter scaling.
Separate Single-d instantiations verify architectural equality at each width.
No trained weights are loaded; latency is an uninstrumented CUDA microbenchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from mwm.adapters.builder import build_mwm_from_stable_config


LEVELS = (48, 72, 96, 120, 144, 168, 192)


class EncoderPath(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(pixels, interpolate_pos_encoding=True)
        return self.projector(encoded.last_hidden_state[:, 0])


class DynamicsStep(nn.Module):
    def __init__(self, transition: nn.Module) -> None:
        super().__init__()
        self.transition = transition

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition.predict(latent, action)


def params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def component_profile(module: nn.Module, inputs: tuple[torch.Tensor, ...], warmup: int, repeats: int) -> dict:
    module = module.eval().to(device="cuda", dtype=torch.float32)
    inputs = tuple(x.to(device="cuda", dtype=torch.float32) for x in inputs)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        output = module(*inputs)
    flops = int(counter.get_total_flops())
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        torch.cuda.synchronize()
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(*inputs)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
    elapsed = sorted(float(start.elapsed_time(end)) for start, end in events)
    return {
        "input_shapes": [list(x.shape) for x in inputs],
        "output_shape": list(output.shape),
        "params": params(module),
        "flops_per_pass": flops,
        "latency_ms": {
            "median": statistics.median(elapsed),
            "p10": elapsed[int((len(elapsed) - 1) * 0.10)],
            "p90": elapsed[int((len(elapsed) - 1) * 0.90)],
            "min": elapsed[0],
            "max": elapsed[-1],
        },
    }


def build(kwargs: dict, levels: tuple[int, ...]) -> nn.Module:
    values = dict(kwargs)
    values["K"] = list(levels)
    return build_mwm_from_stable_config(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--context", type=int, default=1)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert args.warmup > 0 and args.repeats >= 10 and args.context > 0
    torch.manual_seed(3072)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    kwargs = config["kwargs"]
    assert kwargs["K"] == [192]
    assert kwargs["image_shape"] == [224, 224]
    assert int(kwargs["action_dim"]) > 0

    mwm = build(kwargs, LEVELS)
    encoder = EncoderPath(mwm.encoder, mwm.projector)
    output = {
        "protocol": {
            "source_config": str(args.config),
            "checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_config_sha256": kwargs["source_config_sha256"],
            "instantiation": "Saved Single-K=192 base source config; K replaced with seven released MWM levels; no weights loaded",
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "precision": "float32; TF32 disabled",
            "mode": "eval; no_grad FLOP count; inference_mode latency",
            "batch_size": 1,
            "history_length": args.context,
            "action_dim": int(kwargs["action_dim"]),
            "warmup_passes": args.warmup,
            "timed_passes": args.repeats,
            "timing": "Synchronized CUDA-event per forward pass; median; FLOP profiler disabled during timing",
            "flops": "torch.utils.flop_counter.FlopCounterMode supported operator totals for one forward pass",
            "excludes": "preprocessing, transfers, reconstruction decoders, CEM search, episode rollout loop",
        },
        "encoder_path": component_profile(encoder, (torch.rand(1, 3, 224, 224),), args.warmup, args.repeats),
        "mwm_dynamics": {},
        "single_d_dynamics": {},
        "single_d_architecture_check": {},
    }
    for level, transition in zip(LEVELS, mwm.transitions):
        key = str(level)
        latent = torch.randn(1, args.context, level)
        action = torch.randn(1, args.context, int(kwargs["action_dim"]))
        output["mwm_dynamics"][key] = component_profile(
            DynamicsStep(transition), (latent, action), args.warmup, args.repeats
        )
        single = build(kwargs, (level,))
        single_encoder_params = params(single.encoder) + params(single.projector)
        if level == 192:
            output["single_d_encoder_path"] = component_profile(
                EncoderPath(single.encoder, single.projector),
                (torch.rand(1, 3, 224, 224),),
                args.warmup, args.repeats,
            )
        output["single_d_dynamics"][key] = component_profile(
            DynamicsStep(single.transitions[0]), (latent, action), args.warmup, args.repeats
        )
        single_dynamics_params = output["single_d_dynamics"][key]["params"]
        output["single_d_architecture_check"][key] = {
            "encoder_path_params": single_encoder_params,
            "dynamics_params": single_dynamics_params,
            "matches_mwm_component_params": (
                single_encoder_params == output["encoder_path"]["params"]
                and single_dynamics_params == output["mwm_dynamics"][key]["params"]
            ),
            "matches_mwm_dynamics_flops": (
                output["single_d_dynamics"][key]["flops_per_pass"]
                == output["mwm_dynamics"][key]["flops_per_pass"]
            ),
        }
        del single
    assert all(x["matches_mwm_component_params"] for x in output["single_d_architecture_check"].values())
    assert all(x["matches_mwm_dynamics_flops"] for x in output["single_d_architecture_check"].values())
    assert output["single_d_encoder_path"]["params"] == output["encoder_path"]["params"]
    assert output["single_d_encoder_path"]["flops_per_pass"] == output["encoder_path"]["flops_per_pass"]
    output["mwm_encoder_plus_dynamics_params"] = output["encoder_path"]["params"] + sum(
        x["params"] for x in output["mwm_dynamics"].values()
    )
    output["seven_single_d_encoder_plus_dynamics_params"] = sum(
        x["encoder_path_params"] + x["dynamics_params"]
        for x in output["single_d_architecture_check"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
