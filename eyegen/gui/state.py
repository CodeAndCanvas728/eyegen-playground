"""GUI state persistence."""

import json

from eyegen import CONFIG_DIR

GUI_STATE_FILE = CONFIG_DIR / "gui_state.json"


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
