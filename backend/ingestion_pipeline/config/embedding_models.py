"""Loader for config/embedding_models.yaml — the candidate embedding models
under evaluation for this pipeline."""
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).with_name("embedding_models.yaml")


def load_models(roles: list[str] | None = None) -> list[dict]:
    """Return the configured model entries, optionally filtered to those
    tagged with at least one of the given roles (e.g. ["segmentation"])."""
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    models = data["models"]
    if roles:
        models = [m for m in models if set(m["roles"]) & set(roles)]
    return models


def get_current_model() -> dict:
    """Return the model entry currently wired into the pipeline (current: true).

    Change the active model by editing embedding_models.yaml: set the desired
    entry's `current` to true and make sure every other entry's is false."""
    current = [m for m in load_models() if m.get("current")]
    if not current:
        raise ValueError("No model marked current: true in embedding_models.yaml")
    if len(current) > 1:
        names = ", ".join(m["name"] for m in current)
        raise ValueError(f"Multiple models marked current: true in embedding_models.yaml: {names}")
    return current[0]
