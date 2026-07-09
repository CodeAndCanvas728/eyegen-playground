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
- History empty state now includes a "Go Generate" CTA button to navigate to the Home tab
- Close-with-generation-running guard: confirm dialog before closing during active generation

### Changed

- Layout restructured: controls scroll (left) + image preview (right) via QSplitter, capped at 420px
- Width/Height moved from main controls to the Settings tab
- Quantize, saved model path, Bonsai, CoreML, and HF Cache config moved to Settings tab
- Backend hint label now appears on the Home page controls panel
- All QSS rewritten as parameterized theme function for consistent dark/light tokens
- Updated pyproject.toml version from 0.1.2 to 0.2.0

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

### Changed

- QSS rewritten with base-8 padding, `min-heights`, semantic classes, and focus states

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
