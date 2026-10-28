# Changelog

All notable changes to EyeGen are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-28

### Added
- **A model is found wherever it is.** The models root is walked (so a HuggingFace cache layout or a dropped folder under it is found, not only the three conventional backend sub-folders), the HuggingFace hub caches are scanned as roots of their own, and so is the OllamaDiffuser model directory. A revision-hashed snapshot is named after the GGUF it holds; a hub snapshot after the repository it came from.
- **A lone GGUF is completed from the repository it came from.** sd-cli needs the text encoder and VAE that a transformer alone does not carry; the tool knows they live in the GGUF's repository, picks the quantised companion (never a 17 GB full-precision encoder), fetches it on first run, and refuses before sd-cli if the model's files exceed the machine's RAM — refusing to guess when the machine's memory cannot be read, because a guard that opens when it cannot see would never fire.
- **Low Memory mode** checkbox in the GUI and `--offload-to-cpu/--no-offload-to-cpu` on the CLI: sd-cli's offload is now a choice. On by default, it streams weights from RAM to the GPU — slower, smaller peak footprint; off for machines with room.
- **CLI on par with the GUI:** `generate` gained `--negative-prompt`, `--use-t5/--no-t5` and `--offload-to-cpu/--no-offload-to-cpu`, each a per-run override that leaves the config file untouched. The T5 toggle is a config value (`use_t5`) both surfaces honour.
- **The model dropdown leads with the scanned models** (shown by name, path as hover tooltip); the path picker stays behind the Custom entry.
- **Prompt boxes are bigger and resizable:** the Prompt and Negative Prompt fields share a draggable vertical splitter.

### Fixed
- A symlinked GGUF keeps its own name: a HuggingFace cache links a blob whose name says nothing, and the link's name is what is read.
- A lone GGUF is the transformer, not an encoder: a name hint may claim a file as `llm` only when another GGUF is left to be the transformer.
- A lone GGUF whose companions' repository cannot be listed (the machine is offline) says the companions could not be listed and to check the connection, instead of running without its encoder into sd-cli's own error; the specification keeps every field (the `alias` included) when the companions are added.
- A first-run companion download lands in the configured `models_dir` tree even when `EYEGEN_MODELS_DIR` points elsewhere — the two could previously disagree, and the re-validation then said "could not be fetched" for files that were there.
- The GUI honours a configured `models_dir`, and accepts custom model directories (including external volumes) instead of rejecting them as unsafe paths; a custom CoreML bundle path is no longer mangled, and a custom HuggingFace cache location is accepted (traversal still refused).
- `Browse Model…` opens the macOS picker for a file **or** a folder in one panel; pressing Cancel closes it cleanly instead of chaining into a second dialog.
- The GUI opens ready instead of locked to a stale saved model: when the restored model no longer resolves, the first resolvable local model is selected instead.
- A catalogued model is not listed twice.
- The UI: sections separated by space and type rather than bordered boxes, a real canvas surface for the preview, room to breathe for inputs, a proper segmented mode switch, and the QSS theme tokens substitute hyphenated names so the stylesheet parses instead of being dropped.

## [Unreleased]

### Added
- Companion files are picked by how much memory they want: a repository that ships both a BF16 text encoder (17+ GB) and a quantised one now gets the quantised one, so a full-precision encoder can no longer take a 16 GB machine down. A model whose files total more than the machine's RAM is refused with a clear message before sd-cli loads it.
- A lone GGUF in a HuggingFace cache layout is completed from the repository it came from: sd-cli needs the model's text encoder and VAE, and the tool now knows where they live, fetches them on first run, and says so instead of letting sd-cli fail on a transformer alone.
- The models root is walked, so a model anywhere under it is found — a HuggingFace cache layout under the root, a folder the user dropped there — not only the three conventional backend sub-folders. A revision-hashed snapshot directory is named after the GGUF it holds.

### Fixed
- A symlinked GGUF keeps its own name: a HuggingFace cache links a blob whose name says nothing, and the tool now reads the link's name instead of resolving it away.
- A lone GGUF is the transformer, not an encoder: a name hint may claim a file as `llm` only when another GGUF is left to be the transformer, so a single-file model under a `qwen-image`-style name is recognised.
- The HuggingFace hub caches and the OllamaDiffuser model directory are scanned as roots of their own: a model either program downloaded is a model on this machine too. A hub snapshot is named after the repository it came from (``unsloth__flux-3-image-klein-GGUF``), and a catalogued model is not listed twice.
- The GUI opens ready instead of locked to a stale saved model: when the restored model no longer resolves, the first resolvable local model is selected instead.
- A custom CoreML bundle path is no longer mangled by the repository-id normalisation, and a custom HuggingFace cache location is accepted outside the home-roots allow-list (traversal is still refused).
- The UI: sections are separated by space and type rather than bordered boxes, the preview is a real canvas surface, inputs have room to breathe, and the mode switch is a proper segmented control. The QSS theme tokens are substituted for hyphenated names, so the stylesheet parses instead of being dropped.


### Fixed
- GUI now honours a configured `models_dir`, so models stored in the configured folder are recognised.
- GUI and config now accept custom model directories (including external volumes) instead of rejecting them as unsafe paths.

## [0.4.0] - 2026-10-27

### Added
- **A model that is already on this machine can be run by its path.** `./generate.py generate "a fox" --model ~/models/flux1-dev-Q4_K_M.gguf` — what a thing on disk is comes from what is there, not from how it is named: a GGUF file is OllamaDiffuser's own format, a directory laid out the way that package lays one out is a model for it, a directory of weights is a stable-diffusion.cpp model, and a bundle or a diffusers snapshot is CoreML's. The roles of a set of weights are read off the names — the binary takes a path per role, and nothing else says which file is the transformer — and what was decided is printed, so a role the user meant differently is one line of a specification away. A name a catalogue already answers to is suffixed (`klein-2`), so neither model is lost, and a diffusers snapshot is converted to CoreML on its first run; the conversion is kept and reused.
- **A specification can name a file that is on this machine.** A source is either a file in a repository or a path; the stable-diffusion.cpp backend passes a local file where it is, and fetches nothing when every file of the model is already here.
- **`list-models` shows what is on this machine:** the models under the models root, including the ones no catalogue describes — and it still lists them when `models.json` cannot be read, because a broken catalogue is one section's problem, not the listing's.
- `.claude/gate.sh`: a repo gate that runs the project's checks in check mode — ruff check, ruff format --check, a compile of every file CI compiles, the 300-line limit, and pytest — without writing bytecode or caches. CI runs the same script, so the two cannot drift apart. The gate names the missing tool and the command that installs it when it cannot run.
- Dark/light theme system with macOS auto-detection and manual toggle in Settings
- Top navigation bar with Home, History, and Settings tabs
- History tab: thumbnail grid of previously generated images from the output directory
- Collapsible Advanced Settings section (steps, guidance scale, T5 encoder toggle)
- Card-based visual grouping for controls (prompt, model/backend/seed, advanced settings)
- Pill-style mode tab segments for Text-to-Image / Image-to-Image switching
- Rounded-corner thumbnails in History view
- History empty state now includes a "Go Generate" CTA button to navigate to the Home tab
- Close-with-generation-running guard: confirm dialog before closing during active generation

### Changed

- Layout restructured: controls scroll (left) + image preview (right) via QSplitter, capped at 420px
- Width/Height moved from main controls to the Settings tab
- Quantize, saved model path, Bonsai, CoreML, and HF Cache config moved to Settings tab
- Backend hint label now appears on the Home page controls panel
- All QSS rewritten as parameterized theme function for consistent dark/light tokens
- Updated pyproject.toml version from 0.1.2 to 0.2.0
- QSS rewritten with base-8 padding, `min-heights`, semantic classes, and focus states
- **The detection order did not move.** A path is a different question — what a thing on disk is — and not a new rung in the chain of DD-010; a name still resolves exactly as before, and a test pins it.
- The listing no longer stops when the OllamaDiffuser package is missing: what is on this machine does not depend on a package that is not installed.

### Fixed

- `_build_backend_hint` method restored in settings mixin (was orphaned during refactoring)
- **Double-stylesheet cascade:** removed hardcoded inline QSS from `main.py` — all theming now flows through a single QSS source
- **Invisible labels:** replaced 28 inline `setStyleSheet("color:...")` calls with semantic property classes (success/warning/error/hint) resolved by theme QSS
- **Invisible hover feedback:** bumped `surface-hover` contrast from imperceptible (1.56:1 dark, 1.19:1 light) to visible (7.12:1 dark, 3.90:1 light)
- **QTabBar illegible unselected tabs:** changed tab color from `text-muted` to `text-secondary`, restoring AA-level contrast
- **Preview placeholder invisible:** replaced hardcoded `#888` with themed `preview-placeholder` QSS class
- **Button geometry:** standardized all QSS padding to base-8 multiples (7→8, 6→8, 5→8) and added `min-height` to all interactive widgets
- **Layout spacing:** margins 8→16px, vertical spacing 8→16px/12px for consistent section rhythm
- **Window minimum:** reduced from 950×650px to 800×550px
- **Label hierarchy:** added `section-heading` (14px/600w) and `input-label` (13px/500w) classes, applied to 22 structural labels
- **Focus states:** added visible focus indicators to buttons, combos, checkboxes via border-color change (no layout shift)
- **Stop button:** added themed `[class="stop"]` QSS rule replacing hardcoded `#cc3333` inline style
- **MFLUX ran its newer models through the wrong class.** The name-to-class table in `eyegen/_mflux` was written for the families mflux shipped then, and every family added since fell through to `Flux1` — which loads and renders noise, so nothing said it was wrong. mflux's own mapping (`ConfigResolution.resolve_key` → `MODEL_CLASSES`) is asked first now, and the table answers only where it has none: a name its rules cannot place, a key it ships no class for, or the package absent.
- **The transformer of a set of weights is never also an encoder.** A set of two GGUFs whose largest file was named as the text encoder used to give that same file both roles; the transformer is now the largest GGUF no role has claimed.
- **A local model is offered by its path in the GUI.** A name had to be resolved back to the directory it came from, and a catalogue could answer to it first; the path says which model is meant. The same model is no longer listed twice, and a local stable-diffusion.cpp model whose files are where they are is reported installed rather than "fetch it".
- **A relative path is made absolute** before it reaches a runtime or becomes a symlink: a relative path means a different file to each of them.
- **A diffusers snapshot no longer reaches the CoreML subprocess unconverted.** The first run converts it under the CoreML root and reuses that conversion; only a conversion that cannot run (no sidecar, or Apple's converter failing) stops it, with the command that runs the conversion by hand.
- **A converted CoreML bundle runs without the PyTorch model it came from.** The wrapper launched Apple's pipeline with `--model-version`, which called `from_pretrained` and needed the original snapshot; it also treated `-o` as a folder and dropped the randomly chosen seed. The bundle now carries its tokenizer, scheduler and feature extractor in `pipeline_assets/`, a standalone `offline_runner.py` builds Apple's pipeline from those files, writes the image to the exact path it was given, and is passed the seed that names it. A bundle converted by an older version is given the assets without its models being converted again, and a run that could not save the assets keeps what it converted: the next run retries only the assets, and the files replace the old ones only once they are complete. A bundle is never handed another model's tokenizer and scheduler, and a size its models were not exported at is refused by name rather than answered with a differently-sized image. The dead `_offline_wrapper.py` module is deleted.

### Removed

- **`pull-coreml` and the GUI's "Download Model…" button.** Apple's pre-converted repositories keep their models nested (`original/packages`, `split_einsum/packages`) and ship no tokenizer or scheduler, so nothing this version can run was ever produced by that route — and its download is broken under `huggingface_hub` 2.x. `pull --from-hf --backend coreml` refuses with the command that converts instead, so no entry point offers a model that cannot run. Running those bundles is #9.

### Gotchas

- **The tool writes the OllamaDiffuser package's own configuration** (`~/.ollamadiffuser/config.json`) when a GGUF file is registered with it. The package keeps its models in a directory of its own and reads the family off the name of that directory, so the file is *linked* into the directory it expects — the file stays where it is and nothing is copied — and the package is told about the model, so it stays installed for the runs after this one.
- The MLX backend cannot run a local model: its checkpoints are a table inside diffusionkit, and the architecture of each is code there. A diffusers model on disk is CoreML's: the first run converts it (15–20 minutes) and later runs reuse the conversion.

## [0.3.0] - 2026-10-23

### Added
- **Model specifications:** a model is now described by data — the backend that runs it, the files it is made of, the repository each file comes from, and the steps and guidance it was trained for. The user's own models live in `models.json` next to the config file, and the catalogues that ship with the tool are specifications too, so a model the user adds and a model that ships travel the same path.
- **`generate --model`:** use a model for one run without touching the configuration file. The configured default is left as it was.
- **`pull --from-hf`:** turn a HuggingFace repository into a model the tool can run — `./generate.py pull --from-hf owner/repo --as my-model --file diffusion_model=owner/repo:model.gguf --steps 16 --guidance 1.0`. The specification is written to the catalogue and the files are fetched.
- **A model the tool does not know can now run through stable-diffusion.cpp:** the roles a file plays (`diffusion_model`, `vae`, `llm`, `t5xxl`, `clip_l`, `clip_g`, `clip_vision`, `llm_vision`, `tokenizer`, `control_net`) map to the binary's own flags, so any model `sd-cli` can run is one specification away. A model of one file carries one flag.
- **A GGUF model from any repository:** the specification is written to the registry file OllamaDiffuser reads, and the file is fetched from the repository the user named.
- `list-models` lists the models in the user's own catalogue, and the GUI's model dropdown offers them.
- **Every model has a directory of its own:** `models_dir/sdcli/<model>/`, so two models cannot overwrite each other with files of the same name. The root is one setting — `models_dir` in the configuration, or `EYEGEN_MODELS_DIR` for one run — and everything under it follows it.
- The GUI runs the models in the user's own catalogue: the worker carries the specification into the backend, which is what the dropdown has been offering since the catalogue was read.

### Changed
- The stable-diffusion.cpp catalogue is expressed as model specifications, and the backend reads one shape for every model.
- A model name the tool does not know now says where to add it — the catalogue file and `pull --from-hf` — instead of being handed to OllamaDiffuser, which cannot resolve it.
- `validate_saved_model` moved from `eyegen._model_ops` to `eyegen.saved_models`, its own module; the catalogue it reads moved to `eyegen.model_catalogue`.
- **One resolver, both faces:** `resolve_model` is what the CLI and the GUI both ask, so they cannot answer the same question two ways.
- `alias` and `path` in a specification mean something now: an alias becomes the model a fixed-set backend knows, a path becomes the local model it reads.
- The README is up to date with the tool: the real package layout, the whole command list, the auto-detection order, and the families MFLUX 0.23 knows.
- **OllamaDiffuser 2.5.0 works as it is:** every symbol the GGUF path uses — the file the environment variable points at, `model_registry.reload()`, `model_manager.pull_model` / `load_model` / `loaded_model`, `GGUFModelLoader.generate_image` — is where it was in 2.0.17, and the loader's `generate_image` is byte for byte the same. What moved is the package's own model list, from code to a bundled `models.yaml`, and the engine's `strength` default, now `None` where it was 0.75: an editor used to be handed 0.75 whether or not the caller asked, and on FLUX.2 klein, which denoises in four steps, that leaves one step and the model returns confetti. The tool always passes its own strength, so nothing changes for it.
- The floor `update-backends` prints was `>=0.1.0`, a version neither this tool nor its tests have ever seen; it is `>=2.0.17` now, the oldest the registry contract is known to hold on, and a test pins the half nothing covered: a name the package ships becomes the user's when they write it.
- The tests that pin the contract with OllamaDiffuser fail where the package is required, rather than skipping: a green run that verified nothing is worse than a red one.

### Fixed
- **A model's name is checked before it becomes a path.** The name is the directory the model's files are written to, and a name like `../../etc` in `models.json` — or `pull --as ../../x` — would have put them outside the models root. The name is refused where the specification is made, and again where the directory is built; a catalogue the user edited badly says what is wrong rather than raising a traceback.
- **A backend that runs a fixed set of models is refused up front.** `pull --from-hf owner/repo --backend mflux` said the wrong thing — that a stable-diffusion.cpp model needs a file — after the catalogue was written. It now says what is true, before anything is written.
- **The GUI reloaded a model it had already loaded.** The loaded pipeline was kept under a key that did not include the specification, so a catalogue edited for a model that was already loaded kept the old pipeline. The specification is part of the key now.
- **A backend that contradicts a specification is an error** naming both, rather than the specification's backend run in silence. Running the files through a runtime that cannot read them is a wrong picture, not a failure.
- **A guidance of 0 survived:** the code asked whether the setting was there rather than what it was, so a model trained without guidance was given 7.5.
- **A role the backend does not know was dropped silently,** and a model could be fetched for gigabytes and then run with no transformer at all. A role given twice kept the last one; both are refused where the command line is written, and by the backend where a hand-edited catalogue is the input.
- **A file name with a path in it is refused:** the name becomes a path under the model's own directory, and a name like `../../../etc/passwd` would have written outside it.
- **The OllamaDiffuser registry is applied before the package is imported,** everywhere, and only when there are entries. It was applied after the import in two places, and the package reads its table once.
- `status` lists a model the user described; the CoreML hint names the option that exists (`--backend coreml`); the GUI's model dropdown logs a catalogue it cannot read instead of showing an empty list.
- **`save-model` failed with a `TypeError`:** the validation returns a record, which two call sites unpacked as a pair. The bug is older than this branch.
- `./generate.py --version` ran the app rather than printing the version, though the installed `generate` did print it.

### Gotchas
- **The old flat model directory is not read any more.** An install from before this version keeps the stable-diffusion.cpp files in `~/models/eyegen/flux3-klein` and is asked for them in `~/models/eyegen/sdcli/flux-3-image-klein-gguf`. Either re-pull, or move them: `mkdir -p ~/models/eyegen/sdcli/flux-3-image-klein-gguf && mv ~/models/eyegen/flux3-klein/* ~/models/eyegen/sdcli/flux-3-image-klein-gguf/`. The saved MFLUX models moved the same way, from the root into `saved-mflux/`.
- OllamaDiffuser fetches a GGUF's text encoders and VAE for the FLUX and SD 3.5 families only: a GGUF of another family is not runnable through it, and the components are chosen by the family in the model's name.
- The MLX backend cannot run a model that diffusionkit does not know: its checkpoints are a table inside that package, and the architecture of each is code there. A specification for MLX therefore names one of the models diffusionkit has.
- The MFLUX and Bonsai backends run fixed sets of models for the same reason. A specification for them names one of their own models.

## [0.2.1] - 2026-07-07

### Fixed

- **Design system drift:** reconciled all color tokens from purple/cool to oxblood/warm to match the documented design spec (§2F, §3B)
- **§4 Hard Ban #1:** replaced purple-to-pink gradient on primary button with solid oxblood accent
- **WCAG contrast failures:** fixed 3 failing text pairings — dark muted text (3.68:1 → 4.99:1), light hint text (2.71:1 → 4.97:1), disabled button text (was failing, now passing)
- **Spacing scale:** replaced 4 non-base-8 layout values (12px → 16px, 6px → 8px)
- **Typography scale:** aligned font sizes to ≈1.25 modular ratio (11, 13, 16, 20, 24px)

### Changed

- Layout restructured: controls scroll (left) + image preview (right) via QSplitter, capped at 420px
- Width/Height moved from main controls to the Settings tab
- Quantize, saved model path, Bonsai, CoreML, and HF Cache config moved to Settings tab
- Backend hint label now appears on the Home page controls panel
- All QSS rewritten as parameterized theme function for consistent dark/light tokens
- Updated pyproject.toml version from 0.1.2 to 0.2.0

### Fixed

- `_build_backend_hint` method restored in settings mixin (was orphaned during refactoring)
