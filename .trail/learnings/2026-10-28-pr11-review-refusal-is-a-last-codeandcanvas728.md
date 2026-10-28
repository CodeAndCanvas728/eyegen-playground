---
title: "PR11 review: refusal is a last resort; memory guard; native either-or browse picker"
author: CodeAndCanvas728
date: 2026-10-28
id: l-ee02d41e
scope: repo
status: active
src: agent
touches:
  - CHANGELOG.md
  - eyegen/backends/sdcli/install.py
  - eyegen/backends/sdcli/models.py
  - eyegen/backends/sdcli/runner.py
  - eyegen/cli/_generate_helpers.py
  - eyegen/gui/main_window_handlers.py
  - tests/test_backends_sdcli_memory.py
---

Guiding principle (from the owner, vault DD-019): refusal is a last resort in this app. Handle everything possible automatically in the background; the highest friction allowed is an explicit actionable prompt ("download of <xyz> required: Ok? Cancel?"); refuse only when nothing else is possible, and then with the reason stated. Applied here: a lone GGUF whose companions cannot be listed must say so (never run half a model into sd-cli), a first-run fetch goes to the validated tree, and the memory guard fails closed with a clear message instead of inventing an exabyte of RAM. Also: on macOS os.sysconf("SC_PHYS_PAGES") works (returns hw.memsize), so the review claim that it raises ValueError did not reproduce; still used sysctl hw.memsize as the canonical path and made the fallback loud. PySide6 QFileDialog.FileMode.Directory is the native macOS open panel that accepts a file or a folder; the static helpers force ShowDirsOnly and hide files.
