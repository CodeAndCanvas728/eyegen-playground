---
title: "The 300-line gate counts test modules: the sdcli model tests were already at the limit"
author: CodeAndCanvas728
date: 2026-11-02
branch: gitbutler/workspace
id: l-ed1d26f4
scope: repo
status: active
src: agent
touches:
  - AGENTS.md
  - CLAUDE.md
---

## Decided
- The 300-line rule in `.claude/gate.sh` counts every tracked file, test
  modules included. A test module already near the limit fails the gate on
  the addition that should have been a few lines, and the gate is the last
  step before a push.
- New tests for completing a lone GGUF belong in
  `tests/test_backends_sdcli_companions.py`, which is about that job and
  has room. `tests/test_backends_sdcli_models.py` sat at 285 lines and went
  over with two tests.

## Watch out
- Check the line count of a test module before adding to it; the rule is
  mechanical and the gate stops the push.

## In the code
- tests/test_backends_sdcli_companions.py
- tests/test_backends_sdcli_models.py
