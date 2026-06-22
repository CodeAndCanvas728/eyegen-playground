# Session Design Profile

> Active for this session. §4 Hard Bans remain active regardless.
> Run `/design-bible quiz` again to reset.

## Profile

```
╔══════════════════════════════════════════════════╗
║  SESSION DESIGN PROFILE                          ║
║  Project: Unnamed                                ║
╠══════════════════════════════════════════════════╣
║  Aesthetic:    editorial                         ║
║  Motion:       expressive                        ║
║  Color:        monochromatic (oxblood) — base #4A1F2A, warm neutral ║
║  Typography:   §0 defaults (serif display × humanist sans body) ║
║  Constraints:  none                              ║
╠══════════════════════════════════════════════════╣
║  Overrides §0:                                   ║
║    - Aesthetic → editorial                       ║
║    - Motion intensity → expressive (default: restrained) ║
║    - Color register → monochromatic oxblood (default: chromatic contrast w/ free accent) ║
║  Carries over:                                   ║
║    - §0 typography pairing                       ║
║    - Warm neutral ground                         ║
║    - §3B ramp structure / §3D spatial composition ║
║    - §4 hard slop bans (all five)                ║
╚══════════════════════════════════════════════════╝
```

**Dominant visual register:** literary-magazine editorial — oxblood ink on warm paper-stock, serif display at scale, restrained motion elevated to editorial choreography (scroll reveals, image parallax, long-form rhythm).

---

## Active Palette

Generated via `python3 scripts/colortools.py palette "#4A1F2A" --neutral warm --json`.
Source JSON: [`palette.json`](./palette.json) (regenerable from §7A command).

**Rule reminder (§2F):** One hue, fully committed. This palette uses ONE accent hue and ONE neutral temperature. Do not introduce a second accent.

### Accent ramp (oxblood)

| Step  | Hex       | Use                                                 |
|-------|-----------|-----------------------------------------------------|
| 50    | `#F9F5F6` | Lightest tint — paper-stock highlight                |
| 100   | `#EFE1E5` | Tint — hover, soft surface                          |
| 200   | `#E0C3CA` | Light tint — borders on dark, secondary text on light |
| 300   | `#CF9BA8` | Mid-light — secondary actions, focus rings          |
| 400   | `#BD6A80` | Mid — hover state for primary actions               |
| 500   | `#AC4962` | Mid-bold — secondary buttons, link text             |
| 600   | `#853D4F` | Bold — primary buttons, key emphasis                |
| 700   | `#5D2D39` | Deep — pressed state, dark mode primary             |
| **base** | **`#4A1F2A`** | **Locked accent** — display headings, key moments |
| 900   | `#361C23` | Darkest — text on light, deep accent on dark         |

### Neutral ramp (warm)

| Step  | Hex       | Use                                  |
|-------|-----------|--------------------------------------|
| 50    | `#F8F7F7` | Light ground                         |
| 100   | `#EAE8E6` | Card surface (light)                 |
| 200   | `#D6D1CD` | Elevated surface (light)             |
| 300   | `#BCB5AE` | Overlay surface (light)              |
| 400   | `#9F9489` | Mid-neutral                          |
| 500   | `#877A6E` | Secondary text (light mode)          |
| 600   | `#6B6157` | Elevated surface (dark)              |
| 700   | `#4C453E` | Card surface (dark)                  |
| 900   | `#2D2925` | Dark ground                          |

### Surface hierarchy

| Level     | Light      | Dark       |
|-----------|------------|------------|
| ground    | `#F8F7F7`  | `#2D2925`  |
| card      | `#EAE8E6`  | `#4C453E`  |
| elevated  | `#D6D1CD`  | `#6B6157`  |
| overlay   | `#BCB5AE`  | `#877A6E`  |

### Text contrast (engine-verified)

| Surface           | Hex       | Recommended text | Ratio  | Pass (AA 4.5:1) |
|-------------------|-----------|------------------|--------|-----------------|
| ground (dark)     | `#2D2925` | `#F7F3EE`        | 13.06  | ✓               |
| card (dark)       | `#4C453E` | `#F7F3EE`        | 8.53   | ✓               |
| elevated (dark)   | `#6B6157` | `#F7F3EE`        | 5.47   | ✓               |
| overlay (dark)    | `#877A6E` | `#141210`        | 4.48   | ✗ (large/display text only — 3:1) |

Light surfaces inherit from the neutral ramp. Body text on `#F8F7F7` ground: use `#2D2925` (neutral 900) for ≥4.5:1 — recheck via `scripts/colortools.py contrast` if non-neutral pairings are introduced.

---

## Typography (carried from §0)

| Role           | Candidates (free-first)                                      |
|----------------|--------------------------------------------------------------|
| Display/heading | Fraunces, Cormorant Garamond, Playfair Display, DM Serif Display |
| Body serif     | Source Serif 4, Literata, Crimson Pro                        |
| Body sans      | Instrument Sans, Cabinet Grotesk, Plus Jakarta Sans          |
| Mono           | JetBrains Mono, Geist Mono                                   |

**Rules in force:**
- Body line height: 1.55–1.75. Display: 1.05–1.2.
- Body line length: 60–72ch.
- Modular scale: 1.25 or 1.333.
- Display tracking: −0.03em to −0.05em.
- Optical sizing on variable fonts.
- No 1–2 word widows on body paragraphs.

---

## Motion (override: expressive)

§3C default easing curves still apply for UI feedback. Expressive register permits:
- Scroll reveals (choreographed by section, not every element)
- Image parallax (≤20% travel, never on text containers)
- Editorial section transitions (≤500ms)
- Atmospheric ambient loops only within `editorial` register restraint — no breathing glows, no parallax on body copy

| Trigger              | Duration   | Easing                                   |
|----------------------|------------|------------------------------------------|
| Element entrance     | 200–280ms  | `cubic-bezier(0.16, 1, 0.3, 1)` — expo out |
| Feedback             | 100–150ms  | `cubic-bezier(0.34, 1.56, 0.64, 1)` — soft spring |
| State transition     | 150–220ms  | `cubic-bezier(0.16, 1, 0.3, 1)`          |
| Contextual reveal    | 120–180ms  | `cubic-bezier(0.16, 1, 0.3, 1)`          |
| Scroll-choreographed | 400–500ms  | `cubic-bezier(0.16, 1, 0.3, 1)`          |

`prefers-reduced-motion: no-preference` wraps all non-essential animation. GPU only (`transform`, `opacity`).

---

## §4 Hard Slop Bans (always active)

- No purple-to-blue gradient hero sections
- No Inter/Roboto as typographic intention
- No rounded-square icon tiles above section headings
- No skeleton loaders masking genuine slowness
- No ambient glows scattered across multiple elements

---

## Regenerate

```bash
python3 ~/.opencode/skills/design-bible/scripts/colortools.py palette "#4A1F2A" --neutral warm --json > design-system/palette.json
```
