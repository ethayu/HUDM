"""Plot success rate vs. planning FLOPs for the 2D (pop_size, n_iter) grid sweep."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_VALUES = [6, 12, 48, 96, 144, 192]   # MWM K at each level index 0-5
HORIZON    = 5
BUDGET     = 50
REC_HOR    = 5
N_REPLANS  = BUDGET // REC_HOR          # 10 replanning calls per episode


def k_eff_for_scheduler(sched: str, n_iter: int) -> float:
    """Average K across CEM iterations for a given scheduler and n_iter."""
    if sched == "fixed_finest":
        return 192.0
    if sched == "k96_to_finest":
        # Linear ramp from level 3 (K=96) to level 5 (K=192)
        start_l, end_l = 3, 5
        total_k = 0.0
        for i in range(n_iter):
            prog = 0.0 if n_iter <= 1 else i / (n_iter - 1)
            level = int(round(start_l + (end_l - start_l) * prog))
            total_k += K_VALUES[level]
        return total_k / n_iter
    raise ValueError(f"Unknown scheduler: {sched}")


def planning_flops(pop: int, n_iter: int, sched: str) -> float:
    """Analytical FLOPs proxy per episode (full-budget): replans × n_iter × pop × horizon × K_eff."""
    return N_REPLANS * n_iter * pop * HORIZON * k_eff_for_scheduler(sched, n_iter)


def load_sr(subdir: str, env: str, sched: str, pop: int, n_iter: int):
    p = Path(f"rollouts/cem_sweep/{subdir}/{env}/{sched}_pop{pop}_niter{n_iter}/eval.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["swm_results"]["success_rate"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SERIES = [
    ("Dense MWM\nfixed_finest",  "grid_dense_fixed", "fixed_finest",  "#1f77b4", "o"),
    ("Dense MWM\nk96_to_finest", "grid_dense_k96",   "k96_to_finest", "#ff7f0e", "s"),
]

ENV_CONFIGS = {
    "pusht":   {"pop_sizes": [50, 100, 150, 200],  "title": "PushT"},
    "tworoom": {"pop_sizes": [5, 10, 20, 30, 50],  "title": "TwoRoom"},
}
N_ITERS = [5, 10, 20, 30, 50]

OGB_CUBE_SERIES = [
    ("Retrained\nfixed_finest",  "grid_ogb_cube_retrained_fixed", "fixed_finest",  "#1f77b4", "o"),
    ("Dense MWM\nk96_to_finest", "grid_ogb_cube_dense_k96",       "k96_to_finest", "#ff7f0e", "s"),
]
OGB_CUBE_POP_SIZES = [100, 150, 200, 225, 250, 275, 300, 350]
OGB_CUBE_N_ITERS = [5, 10, 20, 30]

# ---------------------------------------------------------------------------
# Plot — one figure per environment
# ---------------------------------------------------------------------------
suptitle = ("Dense MWM — CEM Planning Efficiency: Success Rate vs. FLOPs\n"
            r"(fixed_finest: $K=192$ throughout; k96_to_finest: ramps $K=96\to192$)")

for env, ecfg in ENV_CONFIGS.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    pop_sizes = ecfg["pop_sizes"]

    all_xs = []
    for label, subdir, sched, color, marker in SERIES:
        xs, ys = [], []
        for pop in pop_sizes:
            for n in N_ITERS:
                sr = load_sr(subdir, env, sched, pop, n)
                if sr is None:
                    continue
                xs.append(planning_flops(pop, n, sched))
                ys.append(sr)

        if not xs:
            continue
        all_xs.extend(xs)

        order = np.argsort(xs)
        xs_s = np.array(xs)[order]
        ys_s = np.array(ys)[order]

        ax.scatter(xs_s, ys_s, color=color, marker=marker, s=65,
                   alpha=0.8, zorder=3, label=label.replace("\n", " "))

        # Pareto frontier (running max)
        pareto_x, pareto_y = [xs_s[0]], [ys_s[0]]
        best = ys_s[0]
        for x, y in zip(xs_s[1:], ys_s[1:]):
            if y >= best:
                best = y
                pareto_x.append(x)
                pareto_y.append(y)
        ax.step(pareto_x, pareto_y, where="post", color=color,
                linewidth=2.0, alpha=0.9)

    # Tight x-axis range based on actual data
    xmin, xmax = min(all_xs), max(all_xs)
    pad = 0.3  # log-space padding
    ax.set_xlim(10 ** (np.log10(xmin) - pad), 10 ** (np.log10(xmax) + pad))

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xlabel("Planning FLOPs per episode (proxy)\n"
                  r"$N_{replans} \times n_{iter} \times pop \times H \times K_{eff}$",
                  fontsize=10)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(ecfg["title"], fontsize=14, fontweight="bold")
    ax.set_ylim(max(0, min(all_xs and [load_sr(s[1], env, s[2], p, n)
                for s in SERIES for p in pop_sizes for n in N_ITERS
                if load_sr(s[1], env, s[2], p, n) is not None] or [0]) - 10), 105)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="lower right")
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.tight_layout()

    out = Path(f"rollouts/cem_grid_flops_{env}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")

# ---------------------------------------------------------------------------
# OGBench Cube plot
# ---------------------------------------------------------------------------
ogb_suptitle = ("OGBench Cube — CEM Planning Efficiency: Success Rate vs. FLOPs\n"
                r"(fixed_finest: $K=192$ throughout; k96_to_finest: ramps $K=96\to192$)")

fig, ax = plt.subplots(figsize=(7, 5))
all_xs = []
for label, subdir, sched, color, marker in OGB_CUBE_SERIES:
    xs, ys = [], []
    for pop in OGB_CUBE_POP_SIZES:
        for n in OGB_CUBE_N_ITERS:
            sr = load_sr(subdir, "ogb_cube", sched, pop, n)
            if sr is None:
                continue
            xs.append(planning_flops(pop, n, sched))
            ys.append(sr)

    if not xs:
        continue
    all_xs.extend(xs)

    order = np.argsort(xs)
    xs_s = np.array(xs)[order]
    ys_s = np.array(ys)[order]

    ax.scatter(xs_s, ys_s, color=color, marker=marker, s=65,
               alpha=0.8, zorder=3, label=label.replace("\n", " "))

    pareto_x, pareto_y = [xs_s[0]], [ys_s[0]]
    best = ys_s[0]
    for x, y in zip(xs_s[1:], ys_s[1:]):
        if y >= best:
            best = y
            pareto_x.append(x)
            pareto_y.append(y)
    ax.step(pareto_x, pareto_y, where="post", color=color,
            linewidth=2.0, alpha=0.9)

xmin, xmax = min(all_xs), max(all_xs)
pad = 0.3
ax.set_xlim(10 ** (np.log10(xmin) - pad), 10 ** (np.log10(xmax) + pad))
ax.set_xscale("log")
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax.set_xlabel("Planning FLOPs per episode (proxy)\n"
              r"$N_{replans} \times n_{iter} \times pop \times H \times K_{eff}$",
              fontsize=10)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_title("OGBench Cube", fontsize=14, fontweight="bold")
all_srs = [load_sr(s[1], "ogb_cube", s[2], p, n)
           for s in OGB_CUBE_SERIES for p in OGB_CUBE_POP_SIZES for n in OGB_CUBE_N_ITERS
           if load_sr(s[1], "ogb_cube", s[2], p, n) is not None]
ax.set_ylim(max(0, min(all_srs) - 10), 105)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="lower right")
fig.suptitle(ogb_suptitle, fontsize=10, y=1.02)
fig.tight_layout()

out = Path("rollouts/cem_grid_flops_ogb_cube.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out}")
