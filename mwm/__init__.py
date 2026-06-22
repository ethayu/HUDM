"""Canonical Matryoshka World Models package."""

__all__ = ["ImageNetPreprocess", "MatryoshkaWorldModel", "MWMWorldModel", "TransitionPackage"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from mwm.models import world_model

        return getattr(world_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
