"""Worker entry point configured as the ReSwapper module."""

from .engine import ReSwapperEngine


def create_engine() -> ReSwapperEngine:
    """Construct an uninitialized adapter; worker supplies options via initialize()."""
    return ReSwapperEngine()
