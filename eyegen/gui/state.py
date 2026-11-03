"""GUI state persistence."""

import json
from typing import Optional

from eyegen import CONFIG_DIR
from eyegen.config import Backend

GUI_STATE_FILE = CONFIG_DIR / "gui_state.json"


def backend_value(data) -> str:
    """The backend as it is written down: a name, not an enum.

    The combo holds the backend as a plain string, so reading it as an enum
    raises on the way out — and a state file that cannot be written is a
    state file that keeps the values it had.
    """
    return data.value if isinstance(data, Backend) else str(data)


def load_gui_state() -> dict:
    if GUI_STATE_FILE.exists():
        try:
            with open(GUI_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_gui_state(state: dict):
    GUI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GUI_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _pair(entry) -> Optional[tuple[int, float]]:
    """One saved steps/guidance pair, or nothing when it is not one.

    The file is the user's to edit, so a hand-written entry is read as
    something that may be wrong rather than as something that must work.
    """
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry["steps"]), float(entry["guidance"])
    except (KeyError, TypeError, ValueError):
        return None


def settings_by_model(state: dict) -> dict[str, dict]:
    """The steps and guidance each model was last run with.

    A state file written before models kept their own pair holds one pair
    for the model the window was closed on. That pair belongs to that model,
    not to the next one: a one-step experiment on a small CoreML model must
    not follow the user to a model trained for twenty.
    """
    stored = state.get("settings_by_model")
    if isinstance(stored, dict):
        return {
            str(model): {"steps": pair[0], "guidance": pair[1]}
            for model, entry in stored.items()
            if (pair := _pair(entry)) is not None
        }
    model = str(state.get("model") or "")
    pair = _pair(state) if model else None
    if pair is None:
        return {}
    return {model: {"steps": pair[0], "guidance": pair[1]}}


def settings_for_model(saved: dict, model: str, defaults: tuple[int, float]):
    """The steps and guidance to show for *model*: its own, or *defaults*."""
    entry = saved.get(model)
    if entry is None:
        return defaults
    return entry["steps"], entry["guidance"]
