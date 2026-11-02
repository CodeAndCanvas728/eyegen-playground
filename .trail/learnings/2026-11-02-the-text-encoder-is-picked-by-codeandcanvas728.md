---
title: "The text encoder is picked by the model family; the model's own file is never a candidate"
author: CodeAndCanvas728
date: 2026-11-02
branch: gitbutler/workspace
id: l-b1eb6024
scope: repo
status: active
src: agent
touches:
  - AGENTS.md
  - CHANGELOG.md
  - CLAUDE.md
  - eyegen/backends/sdcli/models.py
  - eyegen/backends/sdcli/pick.py
  - eyegen/backends/sdcli/runner.py
  - tests/test_backends_sdcli_companions.py
  - tests/test_backends_sdcli_pick.py
  - uv.lock
---

## Decided
- The companion encoder is looked up by the *model family* — the repository's
  own name, or the transformer's file name, with the punctuation stripped —
  not by the spelling of the encoder's file name. "qwen3vl_8b" missed
  "qwen3_vl_8b" and "Qwen3-VL-8B"; a repository whose only encoder was
  rejected then ended up with no encoder at all, and the run went on as a
  half model.
- The transformer's own file is never a candidate for the encoder role: a
  transformer called "...-Instruct-....gguf" matches the encoder hints and is
  the slimmest file in the listing, so it was handed to sd-cli as the
  encoder. The scan path already guarded this; the picker needed the same.
- A repository whose only encoder cannot run here (int8/convrot: no Metal
  kernel) is refused, with that file named. A repository with no encoder at
  all is only warned about, and the run continues as before.

## Watch out
- A file name decides more than it should: the same word in a transformer's
  name and in an encoder's name means two different things.

## In the code
- eyegen/backends/sdcli/pick.py: TEXT_ENCODER_COMPANIONS keyed by family,
  _claims_transformer, EncoderChoice.rejected
- eyegen/backends/sdcli/models.py: complete_local_spec refuses on a rejected
  encoder
