#!/usr/bin/env bash
# Repo gate: run by the Claude Code Stop hook when the tree is dirty, and by
# pr-gate before a PR. The checks CI runs (.github/workflows/ci.yml calls this
# script), in check mode: no bytecode, no caches, no file writes.
set -euo pipefail

# The repo root, following symlinks: the caller is external code this repo
# does not control.
script="${BASH_SOURCE[0]}"
while [ -L "$script" ]; do
  dir="$(cd -P -- "$(dirname -- "$script")" && pwd)"
  script="$(readlink -- "$script")"
  case "$script" in /*) ;; *) script="$dir/$script" ;; esac
done
cd "$(cd -P -- "$(dirname -- "$script")/.." && pwd)"

export PYTHONDONTWRITEBYTECODE=1
step() { echo "==> $*"; }

# The tool directory: an explicit override, the active virtualenv, then the
# venvs this repo's setup produces. The first one that has all three tools
# wins; setup.sh's venv/ holds neither ruff nor pytest, so it only counts
# when they are there.
candidates=()
[ -n "${GATE_BIN:-}" ] && candidates+=("$GATE_BIN")
[ -n "${VIRTUAL_ENV:-}" ] && candidates+=("$VIRTUAL_ENV/bin")
candidates+=("$PWD/venv/bin" "$PWD/.venv/bin")
bin=""
for c in "${candidates[@]}"; do
  if [ -x "$c/python" ] && [ -x "$c/ruff" ] && [ -x "$c/pytest" ]; then bin="$c"; break; fi
done
if [ -z "$bin" ]; then
  echo "no python+ruff+pytest in: ${candidates[*]}"
  echo "run 'uv sync --all-extras' to create .venv with ruff and pytest"
  exit 1
fi

step "ruff check";  "$bin/ruff" check --no-cache .
step "ruff format"; "$bin/ruff" format --no-cache --check .

# Every file CI compiles: the entry scripts and all of eyegen/. The list must
# not be empty, or the step would pass without checking anything.
[ -d eyegen ] || { echo "compile: eyegen/ is missing"; exit 1; }
files=()
for f in generate.py gui.py core.py; do [ -f "$f" ] && files+=("$f"); done
while IFS= read -r f; do files+=("$f"); done < <(find eyegen -name '*.py' -type f)
[ "${#files[@]}" -gt 0 ] || { echo "compile: no Python files found"; exit 1; }
step "compile"
"$bin/python" - "${files[@]}" <<'PY'
import sys
from pathlib import Path

for name in sys.argv[1:]:
    compile(Path(name).read_text(encoding="utf-8"), name, "exec")
PY

# The 300-line rule from AGENTS.md II, which only pre-commit enforces.
step "file length"
git rev-parse --git-dir >/dev/null || { echo "file length: not a git repository"; exit 1; }
while IFS= read -r f; do
  case "$f" in *.lock|*.snap|CHANGELOG.md|README.md|*/migrations/*) continue ;; esac
  grep -qI . "$f" || continue  # binary files, which the pre-commit hook skips too
  n=$(wc -l < "$f" | tr -d ' ')
  [ "$n" -gt 300 ] && { echo "$f: $n lines (>300)"; exit 1; }
done < <(git ls-files)

# CI's second job (uv sync + pytest --cov) is not mirrored here.
step pytest; "$bin/pytest" -q -p no:cacheprovider
