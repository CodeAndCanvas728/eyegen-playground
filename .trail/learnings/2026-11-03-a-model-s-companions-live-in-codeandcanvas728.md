---
title: "A model's companions live in a directory of their own, and that directory is not a model"
author: CodeAndCanvas728
date: 2026-11-03
branch: gitbutler/workspace
id: l-331daf7a
scope: repo
status: active
src: agent
touches:
  - AGENTS.md
  - CHANGELOG.md
  - CLAUDE.md
  - eyegen/artifacts.py
  - eyegen/backends/sdcli/pick.py
  - eyegen/backends/sdcli/runner.py
  - tests/test_artifacts.py
  - tests/test_backends_sdcli.py
---

## Decided
- A directory that holds a VAE and a text encoder but **no transformer** is
  where the tool keeps a model's companions, not a model of its own. The
  tool fetches the encoder and the VAE into `sdcli/<model-name>/` while
  the transformer itself sits in the HuggingFace cache under
  `models--<org>--<repo>/snapshots/<revision>/`. The companion directory
  was offered in the model list and run as the model, and sd-cli failed
  with `get sd version from file failed: ''` — an encoder is not a
  diffusion model and has no version to read.
- The rule that caught it is the VAE, not the file count. A directory of
  one GGUF is normally that file's model (a snapshot directory holds one
  transformer and nothing else), so the companion directory has to be
  told apart by what else is in it: a VAE belongs to a model, and a
  directory with a VAE and no transformer is where that model's
  companions were kept.
- The run refuses an encoder handed over as the model, naming the file
  and saying which to choose. The scan fix stops the wrong entry from
  being offered; the refusal stops a path typed by hand, or a stale
  selection, from being run.

## Watch out
- "qwen" is the model's **family**, not the encoder's name. The hint list
  claimed any GGUF with "qwen" in its name as the encoder, so in a
  directory holding both, the transformer `qwen-image-2.1-….gguf` was
  claimed as the encoder and the encoder was run as the model. The
  encoder's own names — `Qwen3VL`, `Qwen2.5-VL` — say which is which;
  the family's name says nothing.
- A file name decides more than it should: the same word in a
  transformer's name and in an encoder's name means two different things.

## Tried and ruled out
- Deciding by the number of GGUF files alone: a snapshot directory holds
  exactly one GGUF and that file *is* the model, so "one GGUF means a
  model" is right for snapshots and wrong for the companion directory.
  The VAE is what tells the two apart.
- Fixing only the picker (`is_encoder_shaped`): the picker chooses the
  encoder among the files of one model; it never sees the choice of
  *which* file is the model. The companion directory was wrong before
  the picker was consulted.

## In the code
- eyegen/artifacts.py: `infer_sdcli_roles` (the `has_vae` rule),
  `_gguf_role` (the Qwen family hint removed, `Qwen3VL`/`Qwen2.5-VL`
  added)
- eyegen/backends/sdcli/pick.py: `is_encoder_shaped`
- eyegen/backends/sdcli/runner.py: `_check_the_transformer_is_a_model`
