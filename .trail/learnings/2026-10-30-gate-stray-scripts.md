---
title: "gate.sh trips over stray untracked tool scripts; check eyegen+tests directly when the gate fails there"
author: CodeAndCanvas728
date: 2026-10-30
id: l-gate-stray-scripts
scope: repo
status: active
src: agent
touches:
  - .claude/gate.sh
---

`.claude/gate.sh` runs `ruff check --no-cache .` over the whole working tree — untracked files included. Stray tool droppings that sit untracked (`scripts/scripts/token_*.py` from a token-vault plugin, etc.) fail the gate with dozens of T201/no-print errors that have nothing to do with the project, locally. CI never sees them (it checks out tracked files only), so a red local gate that is entirely inside untracked dirs is a false alarm — verify with `ruff check --no-cache . 2>&1 | grep -oE '\-\-> [^:]+' | sort -u`: if every hit is outside `eyegen/`, run the checks on the real code instead: `ruff check --no-cache eyegen tests generate.py gui.py`, `ruff format --no-cache --check eyegen tests generate.py gui.py`, then `pytest` (the gate's own remaining steps). The 300-line rule in the gate only counts `git ls-files`, so untracked files never violate it. Do not delete or gitignore the stray dirs without asking — they may be someone's tool state.
