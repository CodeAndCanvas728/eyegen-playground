---
title: "A review fix goes onto the PR's own branch: but commit --branch, then but push"
author: CodeAndCanvas728
date: 2026-11-02
branch: gitbutler/workspace
id: l-05fafdb9
scope: repo
status: active
src: agent
touches:
  - AGENTS.md
  - CLAUDE.md
---

## Decided
- The workspace here holds several stacks at once — the docs stack sits on
  top of the code stack — so a plain `but commit` is not aimed at the
  branch a pull request needs. `but commit --branch <pr-branch> -m ... <files>`
  puts the commit on that branch's tip, and `but push <pr-branch>` sends it.
- A file whose change depends on a commit in another stack cannot be
  committed onto a different branch: `but` refuses with "line N depends on
  <branch>" and names the stack. Leave that file to its own stack — here
  `uv.lock` belongs to fix/app-boot-venv-resilience, so the version sync
  belongs there, not in a review fix on another branch.

## Watch out
- `but commit` takes file paths as arguments; pass every file you mean, or
  it takes the whole dirty tree. This workspace carries unrelated
  uncommitted harness files that must not ride along.

## In the code
- eyegen/backends/sdcli/pick.py, models.py and runner.py (the files the
  review fix touched)
