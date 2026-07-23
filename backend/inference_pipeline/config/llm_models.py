"""Loader for config/llm_models.yaml — the candidate LLM providers/models
under evaluation for answer generation."""
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).with_name("llm_models.yaml")


def load_models() -> list[dict]:
    """Return the configured model entries."""
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return data["models"]


def get_current_model() -> dict:
    """Return the model entry currently wired into the pipeline (current: true).

    Change the active model by editing llm_models.yaml: set the desired
    entry's `current` to true and make sure every other entry's is false."""
    current = [m for m in load_models() if m.get("current")]
    if not current:
        raise ValueError("No model marked current: true in llm_models.yaml")
    if len(current) > 1:
        names = ", ".join(m["name"] for m in current)
        raise ValueError(f"Multiple models marked current: true in llm_models.yaml: {names}")
    return current[0]
