---
title: "A model's steps belong to the model, not to the window"
author: CodeAndCanvas728
date: 2026-11-03
branch: gitbutler/workspace
id: l-b22793e9
scope: repo
status: active
src: agent
touches:
  - eyegen/gui/main_window_state.py
  - eyegen/gui/state.py
  - eyegen/gui/main_window_controls.py
  - eyegen/backends/sdcli/constants.py
---

## Decided
- Steps and guidance are kept per model, in `settings_by_model` in
  `gui_state.json`. A state file written before that table held one pair,
  and it is read as belonging to the model the file was saved with — a
  one-step experiment on a small CoreML model must not follow the user to
  a model trained for twenty.
- A model with no pair of its own is shown what its specification states,
  then what its family states (`SDCLI_FAMILY_SETTINGS`), then the shipped
  catalogue entry, then the tool's defaults. The MFLUX "four steps" rule is
  no longer a separate mechanism with its own remember/restore fields: it
  is just the MFLUX models' default, remembered with the model like any
  other.
- A step count below what a family renders at is raised to the floor, with
  a warning naming the value asked for. One step of a twenty-step flow
  model is not a faster picture: the VAE decodes an almost untouched
  latent and the result is a flat field of colour, not a poor drawing.

## Watch out
- The backend combo holds a plain string, not the `Backend` enum: PySide6
  stores a `str` subclass as a string. Reading `currentData().value` raised
  `AttributeError` at close, and `closeEvent` only caught `ValueError` and
  `OSError`, so **the state file stopped being written** — which is how a
  July state file with `steps: 1` was still the one being read in November.
  A state file that cannot be written keeps the values it had.
- `eyegen/gui/__init__.py` now imports `main` lazily, so the pure helpers
  beside it (state reading and writing) can be tested without PySide6.

## In the code
- eyegen/gui/state.py: `settings_by_model`, `settings_for_model`,
  `backend_value`
- eyegen/gui/main_window_controls.py: `_apply_model_settings`,
  `_model_defaults`
- eyegen/backends/sdcli/constants.py: `SDCLI_FAMILY_SETTINGS`,
  `family_settings`, the floor in `resolve_settings`
