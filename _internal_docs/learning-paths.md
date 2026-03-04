# Learning Paths System — Internal Documentation

## Overview

Learning Paths provide guided sequences through existing posts. Each path is a DAG of posts organized into named parts, defined in YAML, with JavaScript-powered prev/next navigation when a reader follows a path.

**User-facing flow:** Browse `/learning-paths/` listing → click a path → read intro + see part-organized TOC → click "Start this path" → read posts with banner + prev/next nav.

**Build-time flow:** YAML DAG → topological sort → generate .qmd pages + JSON manifest.

---

## Build Pipeline

```
YAML definition (_learning-paths/<id>.yml)
    │
    ▼
generate_learning_paths.py        ← run via make build
    │  Parses YAML, toposorts nodes, extracts post titles
    │
    ├──► learning-paths/<id>.qmd  (per-path page with parts, step cards, Start button)
    └──► assets/learning-paths.json (manifest for runtime JS)
```

Both outputs are gitignored — they're regenerated on every build.

---

## YAML Format

Each path is defined in `_learning-paths/<id>.yml`:

```yaml
title: "Path Title"
description: "One-sentence summary for listing cards."
level: beginner   # beginner | intermediate | advanced (controls sort order)

parts:
  - title: "Part Name"
    description: "What this part covers."
    nodes:
      - slug: post-slug-1
      - slug: post-slug-2
        requires: [post-slug-1]     # DAG edges (can cross part boundaries)
        order: 1                     # optional tie-breaker for toposort
```

### Key rules

- Every `slug` must correspond to an existing post in `posts/`
- `requires` references must be slugs within the same path (can cross parts)
- `order` is optional — used to break ties when multiple nodes have in-degree 0
- Tie-breaking: `(order or 999, slug alphabetical)`

### Optional companion files

| File | Purpose |
|------|---------|
| `_learning-paths/<id>-content.md` | Prose inserted into generated .qmd (intro, learning outcomes, prerequisites) |
| `_learning-paths/<id>-card.png` | Card image for the listing page (copied to `learning-paths/` at build time) |

---

## Generated Outputs

### `learning-paths/<id>.qmd`

Frontmatter includes title, description, subtitle (`"Level · N parts · M posts"`), order (beginner=1, intermediate=2, advanced=3), and optional image.

Body structure:
1. Content from `*-content.md` (if exists)
2. "Start this path" button linking to first post with `?path=<id>&step=1`
3. Part headers with descriptions
4. Step cards (`::: {.lp-step-card}` fenced divs) with step number, linked title, and description

### `assets/learning-paths.json`

```json
{
  "<path-id>": {
    "title": "...",
    "description": "...",
    "level": "beginner",
    "parts": [
      {
        "title": "Part Name",
        "description": "...",
        "steps": [
          {"slug": "...", "title": "...", "description": "...", "url": "/posts/..."}
        ]
      }
    ],
    "steps": [/* flat ordered list of all steps across parts */]
  }
}
```

---

## Runtime JavaScript (`assets/learning-paths.js`)

IIFE pattern (matches `download-bundle.js`). Only activates on `/posts/<slug>/` pages with `?path=<id>` query parameter.

### What it does

1. **Gate**: checks URL matches `/posts/<slug>/` and has `?path=` param
2. **Fetch**: loads `/assets/learning-paths.json`
3. **Find**: locates current post in path steps (by `?step=` param or slug fallback)
4. **Top banner**: injects after `.quarto-title-block` — shows "Part N: Title — Step X of Y in [Path Name]" with prev/next links
5. **Bottom nav**: injects before `#discuss-links` — two-column prev/next cards with part name and post title

All generated links preserve `?path=<id>&step=<n>` params to maintain path context.

### Part awareness

`findPart(parts, globalIdx)` maps a global step index to part number, title, and step-within-part count. Used in both banner text and bottom nav cards.

---

## Styling (`styles/_learning-paths.scss`)

| Class | Purpose |
|-------|---------|
| `.lp-banner` | Top navigation bar (flex, blue-50 background, mono font) |
| `.lp-banner-label` | Path/part/step text in banner |
| `.lp-banner-nav` | Prev/next links in banner |
| `.lp-nav` | Bottom nav container (2-column grid) |
| `.lp-nav-card` | Prev or next card (border, rounded, hover effect) |
| `.lp-nav-direction` | "← Previous" / "Next →" label |
| `.lp-nav-part` | Part name above post title |
| `.lp-nav-title` | Post title in nav card |
| `.lp-step-card` | Step cards on the path page (full-width, border, hover) |

Mobile (≤768px): banner stacks vertically, nav becomes single column.

Dark mode: uses CSS custom properties (`var(--cb-blue-*)`, `var(--color-*)`) — automatic.

---

## Listing Page (`learning-paths.qmd`)

Hand-authored, committed to git. Uses Quarto's `default` listing type with `image-align: left`:

```yaml
listing:
  id: paths
  contents: "learning-paths/"
  type: default
  sort: "order"
  image-align: left
  fields: [image, title, subtitle, description]
```

---

## Wiring

| File | What was added |
|------|----------------|
| `_quarto.yml` | `assets/learning-paths.json` in `project.resources`; "Learning Paths" navbar item (before About); `includes/learning-paths.html` in `include-after-body` |
| `includes/learning-paths.html` | `<script src="/assets/learning-paths.js"></script>` |
| `styles/modern-light.scss` | `@import 'learning-paths'` |
| `styles/modern-dark.scss` | `@import 'learning-paths'` |
| `Makefile` | `LP_SOURCES` variable; `assets/learning-paths.json` target as pre-render dependency |
| `.gitignore` | `learning-paths/*.qmd`, `assets/learning-paths.json` |

---

## Adding a New Learning Path

1. Create `_learning-paths/<new-id>.yml` with the YAML format above
2. (Optional) Create `_learning-paths/<new-id>-content.md` with intro prose, learning outcomes, prerequisites
3. (Optional) Add `_learning-paths/<new-id>-card.png` for listing thumbnail — use a diagram from one of the path's posts
4. Run `make build` — the generator auto-discovers all `.yml` files
5. Verify with `make preview` or deploy

---

## Build Script Details (`_scripts/generate_learning_paths.py`)

### Custom YAML parser

No PyYAML dependency (matches `compile_prompts.py` pattern). Handles our fixed nested format: top-level scalars + `parts:` list with `title`, `description`, `nodes:` sub-list with `slug`, `requires`, `order`.

### Topological sort

Kahn's algorithm across ALL nodes in a path (cross-part requires supported). Tie-breaking by `(order or 999, slug)` tuples in a min-heap. Detects cycles and missing slugs with clear error messages.

### Post metadata extraction

Reads titles and descriptions from:
- Notebook posts: first raw cell of `posts/<slug>/index.ipynb`
- Markdown posts: frontmatter of `posts/<slug>.md`

### URL generation

- `posts/<slug>/index.ipynb` or `posts/<slug>/index.qmd` → `/posts/<slug>/`
- `posts/<slug>.md` → `/posts/<slug>`
