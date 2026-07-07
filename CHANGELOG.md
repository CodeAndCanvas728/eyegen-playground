# Changelog

All notable changes to EyeGen are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dark/light theme system with macOS auto-detection and manual toggle in Settings
- Top navigation bar with Home, History, and Settings tabs
- History tab: thumbnail grid of previously generated images from the output directory
- Collapsible Advanced Settings section (steps, guidance scale, T5 encoder toggle)
- Card-based visual grouping for controls (prompt, model/backend/seed, advanced settings)
- Pill-style mode tab segments for Text-to-Image / Image-to-Image switching
- Rounded-corner thumbnails in History view

### Changed

- Layout restructured: controls scroll (left) + image preview (right) via QSplitter, capped at 420px
- Width/Height moved from main controls to the Settings tab
- Quantize, saved model path, Bonsai, CoreML, and HF Cache config moved to Settings tab
- Backend hint label now appears on the Home page controls panel
- All QSS rewritten as parameterized theme function for consistent dark/light tokens
- Updated pyproject.toml version from 0.1.2 to 0.2.0

### Fixed

- `_build_backend_hint` method restored in settings mixin (was orphaned during refactoring)
