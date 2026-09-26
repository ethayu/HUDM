from __future__ import annotations

def pareto_frontier(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    frontier, best = [], -1.0
    for cost, sr in sorted(pts, key=lambda p: (p[0], -p[1])):
        if sr > best:
            frontier.append((cost, sr))
            best = sr
    return frontier

def cost_to_reach(frontier: list[tuple[float, float]], target: float) -> float | None:
    return next((c for c, s in frontier if s >= target), None)

def success_at(frontier: list[tuple[float, float]], budget: float) -> float | None:
    feasible = [s for c, s in frontier if c <= budget]
    return max(feasible) if feasible else None
