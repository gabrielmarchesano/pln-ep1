"""Resolve pre-reorganization result inputs without filesystem aliases."""
from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_CATEGORIES = (
    Path("research"),
    Path("norberto/delivery"),
    Path("norberto/experiments"),
    Path("norberto/experiments/grid-search"),
)


def resolve_result_input(path: Path) -> Path:
    """Map an existing historical results input to its unique physical location.

    Unknown paths are returned unchanged so callers retain their normal error
    handling. This must only be used for inputs, never new output destinations.
    """
    if path.exists():
        return path
    results_root = REPOSITORY_ROOT / "results" if path.is_absolute() else Path("results")
    try:
        relative = path.relative_to(results_root)
    except ValueError:
        return path
    if len(relative.parts) < 1 or relative.parts[0] in {"research", "norberto"}:
        return path
    matches = [results_root / category / relative for category in RESULT_CATEGORIES
               if (results_root / category / relative).exists()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous historical result path: {path}")
    return matches[0] if matches else path
