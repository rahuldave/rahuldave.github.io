# LLM Explain System — Internal Documentation

## Overview

The LLM Explain feature lets readers use their own Claude API key (stored in browser localStorage) to get AI explanations of educational content. Three interaction modes:

1. **Summarize this page** — full-page summary
2. **Explain up to here** — explain content from the start through a specific point
3. **Explain this code** — explain a specific code block in context

No backend required — the Anthropic API supports CORS directly from browsers via the `anthropic-dangerous-direct-browser-access: true` header.

---

## Architecture: Three Layers

### Layer 1: Lua Filter (`_filters/cell-markers.lua`)

**What it does:** Adds `data-cell-type="code"` attribute to Quarto's existing notebook code cell `<div>` elements.

**Why:** Quarto already wraps code cells in `<div id="cell-N" class="cell">` but markdown cells get unwrapped into bare HTML (sections, paragraphs, etc.). The filter annotates code cells explicitly. It does NOT wrap markdown cells — that approach was considered but rejected because Quarto unwraps markdown cell Divs before user filters run in practice.

**Registration:** Top-level `filters:` in `_quarto.yml` (runs on all formats).

### Layer 2: Python Script (`_scripts/generate_llm_context.py`)

**What it does:** For each post, generates two files in `_site/posts/<slug>/`:

- **`_content.md`** — Clean markdown with cell markers
- **`cells.json`** — Metadata (title, source type, cell list with heading info)

Also generates `_site/llms.txt` — index of all `_content.md` URLs.

**Runs when:** Post-render step. Invoked via `make llm-context` or as part of `make build`.

### Layer 3: Runtime JavaScript (`assets/llm-explain.js`)

**What it does:** IIFE that runs on post pages. Fetches `cells.json`, injects buttons, handles API key management, streams responses from the Anthropic API, renders markdown+LaTeX.

**Loaded via:** `includes/llm-explain.html` contains a single `<script src="/assets/llm-explain.js"></script>` tag, registered as `include-after-body` in `_quarto.yml`. Quarto resolves the path to a relative URL per page. The JS is loaded once and cached by the browser across all pages.

---

## File Inventory

| File | Purpose |
|------|---------|
| `_filters/cell-markers.lua` | Adds `data-cell-type="code"` to notebook cell divs |
| `_scripts/generate_llm_context.py` | Generates _content.md, cells.json, llms.txt |
| `_llm-config.yml` | Human-editable config: model, max_tokens, prompt templates |
| `_scripts/compile_prompts.py` | Compiles `_llm-config.yml` → `assets/llm-prompts.json` |
| `assets/llm-prompts.json` | Generated JSON consumed by JS at runtime (committed to repo) |
| `assets/llm-explain.js` | Runtime JS (buttons, modals, API calls) — cacheable external file |
| `includes/llm-explain.html` | `<script src>` tag that loads `llm-explain.js` |
| `styles/_llm-explain.scss` | Styles (buttons, modals) using CSS custom properties |
| `_quarto.yml` | Registers filter + include-after-body |
| `Makefile` | `llm-context` target + integrated into `build` |

### Why `_content.md` (not `content.md`)

The LLM context file is named with a leading underscore because Quarto skips files and directories prefixed with `_` during rendering. Without the underscore, `content.md` files in `docs/posts/*/` were picked up by Quarto as renderable source files on subsequent builds, causing:
- Duplicate/broken listing items (163 instead of 57) with no dates or categories
- Broken links pointing to non-existent `content.html` pages
- A cascading `docs/docs/` nesting problem from Quarto rendering the `docs/` directory

The `cells.json` file does not need an underscore because `.json` is not a Quarto source type.

### Flat `.md` Posts and LLM Support

Top-level markdown posts (e.g., `posts/boxloop.md`) render to flat URLs like `/posts/boxloop.html`, not to a directory with `index.html`. However, the LLM context files are always written to a directory: `/posts/boxloop/_content.md` and `/posts/boxloop/cells.json`. The JS extracts the slug from the URL (`boxloop`) and constructs `baseUrl = '/posts/' + slug + '/'`, so it correctly fetches `/posts/boxloop/cells.json` — which exists alongside the flat HTML file. Flat `.md` posts are fully supported for LLM interrogation.

---

## _content.md Format

Unified marker format for all post types:

```markdown
<!-- cell:0 type:markdown -->
## What is probability?
Content here...

<!-- cell:1 type:code -->
```python
import numpy as np
x = np.array([1, 2, 3])
```
Output:
```
array([1, 2, 3])
```

<!-- cell:2 type:markdown -->
The code above shows...
```

**Rules:**
- Cell indices are sequential within the file
- For notebooks (`.ipynb`): indices match the original notebook cell indices (0-based), which also match Quarto's `id="cell-N"` in HTML
- For markdown/qmd: indices are assigned sequentially as the parser encounters headings and code blocks
- Code outputs: text stdout is included; images become `[Figure]` or `[Figure: caption]`
- LaTeX math is preserved as-is (Claude reads it natively)

## cells.json Format

```json
{
  "version": 1,
  "source_type": "ipynb",       // or "md" or "qmd"
  "title": "Probability",
  "cells": [
    {"cell": 0, "type": "raw", "skip": true},
    {"cell": 1, "type": "code", "html_id": "cell-1"},
    {"cell": 2, "type": "markdown", "headings": [
      {"slug": "what-is-probability", "level": 2},
      {"slug": "probability-from-symmetry", "level": 3}
    ]},
    {"cell": 4, "type": "code", "lang": "python"}
  ]
}
```

**Fields:**
- `source_type`: Determines button injection strategy in JS
- `cells[].html_id`: Only for notebook code cells (matches `id` in rendered HTML)
- `cells[].headings`: Heading slugs for mapping DOM headings to cell indices
- `cells[].lang`: Only for md/qmd code cells (the language tag)

---

## How Post Types Are Handled

### Notebook posts (`.ipynb`)

- **Source:** `posts/<slug>/index.ipynb`
- **HTML structure:** Code cells → `<div id="cell-N" class="cell">`, markdown cells → bare `<section>`/`<p>`
- **Buttons:** After every `div.cell`: "Explain up to here" + "Explain this code"
- **Content slicing:** Uses `<!-- cell:N -->` markers; N matches Quarto's cell IDs

### Flat markdown posts (`.md`)

- **Source:** `posts/<name>.md` or `posts/<name>/index.md`
- **HTML structure:** Headings → `<section id="slug"><h2>`, code blocks → `<div class="sourceCode">` (no `div.cell` wrapper)
- **Buttons:**
  - After h2/h3 headings: "Explain up to here"
  - After standalone `div.sourceCode` blocks: "Explain up to here" + "Explain this code"
- **Content slicing:** Same `<!-- cell:N -->` markers; indices are sequential from the parser

### QMD posts (`.qmd`)

- **Source:** `posts/<slug>/index.qmd`
- **Executable code blocks** (`{python}`): Rendered identically to notebook code cells → `div.cell` wrappers → handled like notebooks
- **Static code blocks** (` ```python `): Rendered as `div.sourceCode` → handled like markdown code blocks
- **JS-heavy interactive posts** (like `seasons/`): No code blocks in rendered HTML → only "Summarize" button

### Directory `index.md` posts

- **Examples:** `gibbsfromMH/index.md`, `hmcidea/index.md`
- **Previously missed:** Script now checks for `index.md` in directories (alongside `.ipynb` and `.qmd`)
- **Handled identically** to flat `.md` posts

---

## JS Button Injection Logic

```
injectButtons()
├── injectSummarizeButton()          // Always: after #title-block-header
└── injectCellButtons()
    ├── [notebook cells]              // div.cell[id^="cell-"] → "Explain up to here" + "Explain this code"
    └── [non-notebook / fallback]     // When source_type != "ipynb" OR no div.cell found
        ├── h2/h3 headings           // data-anchor-id lookup in cells.json → "Explain up to here"
        └── div.sourceCode           // Standalone (not inside div.cell) → both buttons
```

**Heading → cell index mapping:**
1. cells.json contains `headings: [{slug, level}]` per cell
2. JS builds `slugToCellIndex` lookup: heading slug → cell index
3. Quarto puts heading ID on `<section id="...">` and `data-anchor-id` on the `<h2>`/`<h3>`
4. JS reads `data-anchor-id` and looks up the cell index
5. Fallback: if slug not found, uses DOM order index

---

## Content Slicing (JS)

All post types use the same `<!-- cell:N -->` markers.

| Action | Slicing logic |
|--------|---------------|
| Summarize | Send all of `_content.md` |
| Explain up to here (cell N) | Send from start through end of cell N's content |
| Explain this code (cell N) | Send cell N's code + up to 3000 chars preceding context |

---

## Externalized Config

All LLM settings (model, max tokens, prompts) are defined in `_llm-config.yml` (project root) and compiled to `assets/llm-prompts.json` for runtime use. This lets you change the model or edit prompts without touching JavaScript.

### Workflow

1. Edit `_llm-config.yml`
2. Run `python3 _scripts/compile_prompts.py` (or `make build` — the Makefile rule triggers automatically if the YAML changed)
3. The generated `assets/llm-prompts.json` is committed to the repo so Quarto copies it to `_site/assets/` during render

### _llm-config.yml Format

```yaml
model: claude-sonnet-4-6
max_tokens: 2048

system: >
  Teaching assistant persona...

summarize: >
  Prompt with {title} placeholder...

explain_upto: >
  Prompt with {title} placeholder...

explain_code: >
  Line-by-line explanation prompt...
```

- `model` and `max_tokens` are plain scalars controlling the API call
- Prompts use YAML folded block scalars (`>`) so multi-line text is collapsed to single strings
- `{title}` is a placeholder replaced at runtime by the JS with the post title from `cells.json`
- `system` goes in the API `system` field; the others are user message prefixes

### compile_prompts.py

Small script (~45 lines) that parses the simple YAML without PyYAML (uses regex for both plain scalars and folded block format) and writes JSON. No external dependencies.

### Runtime Behavior (JS)

On page load, `llm-explain.js` fetches `/assets/llm-prompts.json`. If the fetch succeeds, the external config overrides the hardcoded defaults. If it fails (404, network error), hardcoded fallback values are used — so the feature degrades gracefully.

The `buildUserPrompt()` function maps action names to prompt keys (`explain-code` → `explain_code`) and applies `{title}` replacement.

---

## API Integration

- **Model:** Loaded from `config.model` (externalized in `_llm-config.yml`); default `claude-sonnet-4-6`
- **Max tokens:** Loaded from `config.max_tokens`; default `2048`
- **Streaming:** SSE via fetch + ReadableStream
- **System prompt:** Loaded from `config.system` (externalized); falls back to hardcoded default
- **API key storage:** `localStorage` key `claude-api-key`
- **CORS header:** `anthropic-dangerous-direct-browser-access: true`

### Error handling:
- 401 → clears stored key, prompts for new one
- Network error → shows error in chat modal
- Stream interruption → renders partial response
- Missing MathJax → renders raw LaTeX strings (safe)

---

## Edge Cases — Known and Handled

| Edge case | Status | Behavior |
|-----------|--------|----------|
| Empty code cells (notebook) | Handled | Button injected, empty code sent — Claude handles gracefully |
| No cells.json (404) | Handled | Feature silently disabled for that post |
| No _content.md (404) | Handled | Error shown in chat modal |
| Invalid API key (401) | Handled | Key cleared, re-prompt shown |
| Stream interrupted | Handled | Partial text rendered |
| MathJax not loaded | Handled | Raw LaTeX shown (no crash) |
| Listing pages (/posts/) | Handled | Feature not activated |
| Non-post pages (/, /about) | Handled | Feature not activated |
| Posts with no title | Handled | Falls back to slug |
| QMD with only JS (no Python) | Handled | Only "Summarize" button (no code cells) |
| Directory index.md posts | Handled | Parsed as markdown |
| Fenced code blocks in .md | Handled | Extracted as code cells |
| Heading slug mismatch | Handled | Falls back to DOM order |

---

## Build Pipeline

```
make build
  ├── compile_prompts.py      # _prompts.yml → assets/llm-prompts.json (if YAML changed)
  ├── quarto render           # Renders all posts to _site/ (copies llm-prompts.json too)
  │   └── cell-markers.lua    # Adds data-cell-type to code cells
  ├── generate_llm_context.py # Writes _content.md + cells.json to _site/
  └── generate_bundles.py     # Writes zip bundles to _site/
```

The Makefile dependency chain: `_prompts.yml` → `assets/llm-prompts.json` → render stamp → LLM context stamp → bundles stamp.

For development: `make preview` for live reload. After editing `_prompts.yml`, run `python3 _scripts/compile_prompts.py` to update the JSON — the preview server will pick it up.

---

## SCSS Architecture

`styles/_llm-explain.scss` is a partial imported by both `modern-light.scss` and `modern-dark.scss`.

Uses **only CSS custom properties** (`var(--color-surface)`, `var(--cb-blue-300)`, etc.) so the same styles adapt automatically to light/dark mode. No SCSS variables or mode-specific rules.

Key classes:
- `.llm-buttons` — button container (flex, right-aligned, hidden until hover)
- `.llm-btn` — pill-shaped mono-font button
- `.llm-modal-overlay` — fixed backdrop with blur
- `.llm-modal` — card-style modal
- `.llm-key-modal` — API key entry form
- `.llm-chat-modal` — streaming response display
- `.llm-chat-response.llm-streaming` — pre-wrap text during streaming
- `.llm-chat-response` (after render) — full markdown styles

Buttons are **invisible by default** (`opacity: 0`) and reveal on hover of the preceding element (`.cell:hover + .llm-buttons`, `section:hover + .llm-buttons`, etc.).
