# Build & Deploy System — Internal Documentation

## Overview

The site is a Quarto website rendered to `_site/`, with post-render Python scripts that generate LLM context files and downloadable bundles. Deployment pushes `_site/` to the `gh-pages` branch via git worktree. GitHub Pages serves from `gh-pages` (root `/`).

All build orchestration is in the **Makefile**. Stamp files in `_site/` track what's been built so incremental builds skip unchanged stages.

---

## Dependency Graph

```
_llm-config.yml ─────────────────────┐
_scripts/compile_prompts.py ──────┤
                                  ▼
                        assets/llm-prompts.json
                                  │
posts/**/*.ipynb ─────────────┐   │
posts/**/*.qmd ───────────────┤   │
posts/**/*.md ────────────────┤   │
*.qmd, *.md ──────────────────┤   │
_quarto.yml ──────────────────┤   │
styles/*.scss ────────────────┤   │
_filters/*.lua ───────────────┤   │
includes/*.html ──────────────┤   │
assets/*.js ──────────────────┤   │
                              ▼   ▼
                     ┌─────────────────────┐
                     │   quarto render      │  → _site/.stamp.render
                     │   (+ cell-markers    │
                     │     Lua filter)      │
                     └─────────┬───────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │ generate_llm_context │  → _site/.stamp.llm-context
                     │  _content.md         │
                     │  cells.json          │
                     │  llms.txt            │
                     └─────────┬───────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │  generate_bundles    │  → _site/.stamp.bundles
                     │  <slug>.zip          │
                     │  bundles.json        │
                     └─────────┬───────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │  deploy (rsync to    │
                     │  gh-pages branch)    │
                     └─────────────────────┘
```

---

## Make Targets

### Phony Targets (always run when invoked)

| Target | Command | What it does |
|--------|---------|--------------|
| `make preview` | `quarto preview` | Live dev server with hot reload. Does **not** run post-render scripts (LLM context, bundles). |
| `make build` | (depends on bundles stamp) | Full build: compile prompts → render → LLM context → bundles. Only re-runs stages whose inputs changed. |
| `make deploy` | (depends on `build`) | Runs `build`, then rsyncs `_site/` to `gh-pages` branch and pushes. |
| `make test-bundles` | (depends on bundles stamp) | Runs `test_bundles.py` — executes each bundle via `uvx juv exec` and writes `_site/test-report.json`. |
| `make clean` | `rm -f _site/.stamp.*` | Deletes stamp files, forcing a full rebuild on the next `make build`. Does **not** delete `_site/` itself. |

### File Targets (skip if up-to-date)

| Target | Stamp file | Depends on | Script |
|--------|-----------|------------|--------|
| `assets/llm-prompts.json` | (none — real file) | `_llm-config.yml`, `_scripts/compile_prompts.py` | `compile_prompts.py` |
| Quarto render | `_site/.stamp.render` | All `.ipynb`/`.qmd`/`.md` sources, `_quarto.yml`, SCSS, Lua filters, HTML includes, `assets/*.js`, `assets/llm-prompts.json` | `quarto render` + `rm -f _site/CLAUDE.html` |
| LLM context | `_site/.stamp.llm-context` | Render stamp, `generate_llm_context.py`, all `index.ipynb` files | `generate_llm_context.py` |
| Bundles | `_site/.stamp.bundles` | LLM stamp, `generate_bundles.py`, all `index.ipynb` files, all `assets/`+`data/` under posts | `generate_bundles.py` |

---

## Stamp File Mechanism

Each build stage writes a stamp file (e.g., `_site/.stamp.render`) on success. Make compares the stamp's mtime against inputs:

- If any input is newer → stage re-runs
- If stamp is newer than all inputs → stage is skipped

Stamps live inside `_site/` so that `quarto render` (which recreates `_site/`) naturally invalidates them. The `make clean` target deletes stamps without touching rendered output.

The `deploy` target's rsync excludes `.stamp.*` files so they don't end up on `gh-pages`.

---

## Stage Details

### 1. Compile Prompts

```
_llm-config.yml → python3 _scripts/compile_prompts.py → assets/llm-prompts.json
```

- Reads YAML, writes JSON. No external dependencies (parses folded block scalars with regex, no PyYAML).
- Output is committed to the repo. Quarto copies it to `_site/assets/` during render because it's listed in `project: resources:` in `_quarto.yml`.
- Can also be run standalone: `python3 _scripts/compile_prompts.py`

### 2. Quarto Render

```
quarto render → _site/
```

- Renders all `.ipynb`, `.qmd`, and `.md` files to HTML in `_site/`.
- Runs the `_filters/cell-markers.lua` Lua filter (adds `data-cell-type` attributes to code cell divs).
- Copies referenced assets, plus anything in `project: resources:`, to `_site/`.
- `CNAME` and `.nojekyll` in the project root are copied to `_site/` automatically by Quarto.
- Post-render cleanup: `rm -f _site/CLAUDE.html` (CLAUDE.md is rendered but shouldn't be deployed).

**Quarto config** (`_quarto.yml`):
- Site type: `website`
- Explicit resources: `assets/llm-prompts.json` (not discovered via document references)
- Lua filter: `_filters/cell-markers.lua`
- SCSS themes: `styles/modern-light.scss`, `styles/modern-dark.scss`
- HTML includes: fonts, brand icon, discuss links, LLM explain, download bundle, theme toggle

### 3. LLM Context Generation

```
python3 _scripts/generate_llm_context.py
```

- Walks `_site/posts/`, finds rendered posts.
- For each post, generates:
  - `_content.md` — clean markdown with `<!-- cell:N type:T -->` markers
  - `cells.json` — metadata (title, source type, cell list with heading slugs)
- Also generates `_site/llms.txt` — index of all `_content.md` URLs.
- See `_internal_docs/llm-explain-system.md` for format details.

### 4. Bundle Generation

```
python3 _scripts/generate_bundles.py
```

- For each notebook post, creates `<slug>.zip` containing:
  - `index.ipynb` (with PEP 723 dependency cell)
  - `assets/` (images, per-notebook data)
  - `data/` (CSV files, if present)
  - `README.md` (title, link, quick-start instructions)
- Writes `_site/bundles.json` manifest (used by `download-bundle.js` at runtime).

### 5. Deploy

```
make deploy
```

Mechanism:
1. Runs `build` (if not already up-to-date)
2. `git worktree add /tmp/gh-pages-deploy gh-pages` — checks out `gh-pages` into a temp directory
3. `rsync -av --delete --exclude='.git' --exclude='.stamp.*' _site/ /tmp/gh-pages-deploy/` — mirrors `_site/` to the worktree
4. Commits all changes (`--allow-empty` in case nothing changed) and pushes `gh-pages`
5. Removes the temporary worktree

**Important:** `deploy` only pushes `gh-pages`. You must `git push origin main` separately to push source changes.

---

## GitHub Pages Configuration

- Serves from the **`gh-pages` branch**, root `/`
- `CNAME` file contains `rahuldave.com` (custom domain)
- `.nojekyll` disables Jekyll processing (Quarto handles everything)
- Both files live in the project root; Quarto copies them to `_site/`

---

## Scripts Inventory

| Script | Used by | Purpose |
|--------|---------|---------|
| `compile_prompts.py` | Makefile (prompts target) | YAML → JSON prompt compilation |
| `generate_llm_context.py` | Makefile (LLM stamp) | Post-render: `_content.md` + `cells.json` per post |
| `generate_bundles.py` | Makefile (bundles stamp) | Post-render: zip bundles + `bundles.json` manifest |
| `test_bundles.py` | `make test-bundles` | Runs each bundle via `uvx juv exec`, writes test report |
| `import_notebook.py` | `/import-wiki-notes` skill | Imports wiki `.ipynb` as Quarto blog post |
| `inject_juv_metadata.py` | `/bundle-post` skill | Injects PEP 723 dependency cell into notebooks |
| `update_captions.py` | `/caption-images` skill | Updates image captions in notebook markdown cells |
| `add_notebook_format.py` | (legacy, unused) | Adds `format: {html, ipynb}` to notebook frontmatter |
| `clean_toc_keywords.py` | (legacy, unused) | Removes leftover TOC/Keywords cells from imports |

---

## Common Workflows

### Full build and deploy
```bash
make deploy              # build (if needed) + push to gh-pages
git push origin main     # push source separately
```

### Rebuild from scratch
```bash
make clean && make build
```

### Preview during development
```bash
make preview             # live reload; LLM/bundle features won't work
```

To test LLM explain or download bundles locally, stop preview and run `make build`, then serve `_site/` with any static server.

### Edit LLM config (model, prompts)
```bash
# edit _llm-config.yml (model, max_tokens, prompts)
python3 _scripts/compile_prompts.py   # or just make build
```

### Render a single post
```bash
quarto render posts/probability/index.ipynb
```
Note: this only renders the HTML. LLM context and bundles require the full `make build`.

### Test bundles
```bash
make test-bundles        # builds first if needed, then runs juv exec on each
```

---

## Gitignore

Files excluded from `main` branch tracking:

| Pattern | Why |
|---------|-----|
| `_site/` | Build output — deployed via `gh-pages`, not committed to `main` |
| `/.quarto/` | Quarto cache/temp files |
| `.pixi`, `*.egg-info` | Python environment artifacts |
| `**/*.quarto_ipynb` | Quarto's intermediate notebook copies |
| `designs/` | Design mockups, not part of the site |
| `.DS_Store` | macOS metadata |
| `__pycache__/` | Python bytecode cache |
| `.claude/settings.local.json` | Local Claude Code settings |
| `.claude/projects/` | Claude Code project-specific data |

---

## Gotchas

- **`_site/` is not committed to `main`** — it was accidentally tracked at one point and was removed. If it reappears in `git status`, check that `.gitignore` has `_site/` and run `git rm -r --cached _site/`.
- **`make deploy` does not push `main`** — always `git push origin main` separately.
- **`make preview` skips post-render scripts** — LLM explain buttons and download bundles won't appear during preview. Use `make build` + a static server for full testing.
- **CLAUDE.html cleanup** — `CLAUDE.md` is a valid Quarto source file, so `quarto render` produces `_site/CLAUDE.html`. The Makefile deletes it after each render.
- **`assets/llm-prompts.json` needs explicit `resources:`** — Quarto only auto-copies assets discovered through document references. Since the JSON is fetched by JS at runtime, it must be listed in `project: resources:` in `_quarto.yml`.
- **Stamp files inside `_site/`** — `quarto render` recreates `_site/`, which deletes existing stamps, so a render always triggers downstream stages. `make clean` only deletes stamps, not the rendered site.
- **`--allow-empty` on deploy commit** — the deploy step uses `git commit --allow-empty` so it succeeds even if the only changes are timestamps (rsync may touch files without content changes).
