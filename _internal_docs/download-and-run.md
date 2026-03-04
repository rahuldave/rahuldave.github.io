# Download and Run System — Internal Documentation

## Overview

Each notebook post gets a downloadable `<slug>.zip` that readers run locally with `uvx juv run index.ipynb`. The PEP 723 inline script metadata cell tells `juv` exactly which packages to install — no pre-built environment needed.

**User-facing flow:** Click "Download and Run" → unzip → `uvx juv run index.ipynb` → done.

**Build-time flow:** Inject PEP 723 deps → `quarto render` → generate zip bundles + manifest.

---

## Build Pipeline

```
Source notebook (posts/<slug>/index.ipynb)
    │
    ▼
inject_juv_metadata.py          ← run manually via /bundle-post skill
    │  Scans imports, injects PEP 723 cell at index 1
    │  Marks pyodide_compatible = true/false
    ▼
Modified notebook (PEP 723 cell added, committed to git)
    │
    ▼
make build
    │  quarto render  →  _site/posts/<slug>/index.html
    │  ...
    │  generate_bundles.py  →  _site/posts/<slug>/<slug>.zip
    │                       →  _site/bundles.json
    ▼
Deployed via make deploy → gh-pages
```

---

## Step 1: PEP 723 Injection (`_scripts/inject_juv_metadata.py`)

This runs **once per notebook** when first imported (via `/bundle-post` skill), not at every build. The result is committed to git.

### What it does

1. **Scans all code cells** for `import` and `from X import` statements using AST parsing
2. **Maps import names to PyPI packages** (e.g., `sklearn` → `scikit-learn`, `cv2` → `opencv-python`) via `IMPORT_TO_PYPI` table
3. **Filters out** stdlib modules, Jupyter built-ins, and sub-packages (e.g., `scipy.stats` doesn't add a separate dep if `scipy` is already present) via `SKIP_MODULES`
4. **Checks Pyodide compatibility** — packages in `pyodide_incompatible` set (`torch`, `pymc3`, `theano-pymc`, `theano`, `pymc`, `lxml`) flag the notebook as browser-incompatible
5. **Injects a hidden code cell** at index 1 (after frontmatter raw cell at index 0)

### PEP 723 cell format

```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "seaborn",
# ]
# ///
```

- `#| include: false` — hides the cell from Quarto's rendered HTML
- `# /// script` / `# ///` — PEP 723 markers that `juv` reads
- Dependencies are sorted alphabetically

### Idempotency

Re-running the script on a notebook that already has a PEP 723 cell removes the old one first, then re-injects. Safe to re-run after adding new imports.

### Also removes `ipynb: default`

The script strips any `ipynb: default` from the frontmatter YAML (the old Quarto "download notebook" button is replaced by zip bundles).

---

## Step 2: Bundle Generation (`_scripts/generate_bundles.py`)

Runs at build time via `make build` (the `$(BUNDLES_STAMP)` target).

### What it does

For each `posts/<slug>/index.ipynb`:

1. **Skips** notebooks with no real Python code (markdown-only posts)
2. **Collects files** for the zip:
   - `index.ipynb` (the notebook itself, with PEP 723 cell)
   - `assets/*` — images (PNG, JPG, GIF, SVG, WebP) and data (CSV, TSV, JSON, NPY, pickle)
   - `data/*` — same extensions
   - `README.md` — generated with title, link to online version, quick-start instructions, dep list
3. **Creates** `_site/posts/<slug>/<slug>.zip` preserving directory structure
4. **Writes** `_site/bundles.json` manifest

### Manifest format (`bundles.json`)

```json
{
  "probability": {
    "zip": "/posts/probability/probability.zip",
    "size_bytes": 45230,
    "dependencies": ["numpy", "matplotlib"],
    "pyodide_compatible": true,
    "files": ["index.ipynb", "assets/venn.png"]
  }
}
```

### Pyodide compatibility

Hardcoded incompatible set at line ~179:
```python
pyodide_incompatible = {"torch", "pymc3", "theano-pymc", "theano", "pymc", "lxml"}
```

If any notebook dependency is in this set → `pyodide_compatible: false` → no "Run in Browser" button.

**Current incompatible notebooks (6):** `switchpoint`, `utilityorrisk` (pymc), `votingforcongress` (lxml), `functorch`, `mlp_classification`, `nnreg` (torch).

---

## Step 3: Button Injection (`assets/download-bundle.js`)

Runs in the reader's browser on every post page.

1. Matches URL `/posts/<slug>/`
2. Fetches `HEAD /posts/<slug>/<slug>.zip` (checks existence + gets size) and `GET /bundles.json` in parallel via `Promise.all`
3. If zip exists: injects "Download and Run" button with file size
4. If `bundles.json` says `pyodide_compatible: true`: also injects "Run in Browser" button with `target="_blank"` (opens in new tab; see run-in-browser.md)
5. Buttons are inserted into `.llm-summarize-wrap` (the shared button bar)
6. On mobile (≤ 768px), the button bar uses a 2×2 CSS grid layout (styled in `_llm-explain.scss`)

### Button order (left to right)

`[Run in Browser]` `[Download and Run · 164 KB]` `[Download md]` `[Summarize]`

---

## Step 4: Testing (`_scripts/test_bundles.py`)

Run via `make test-bundles`.

For each zip in `_site/posts/*/`:
1. Unzips to a temp directory
2. Runs `uvx juv exec index.ipynb` with timeout (default 300s)
3. Reports pass/fail/timeout/error
4. Writes JSON report to `_site/test-report.json`

**Known slow notebooks** (need `--timeout 1200`): `samplingclt` (~230s), `switchpoint` (~90s), `mlp_classification` (~90s), `nnreg` (~80s).

All 49 bundles pass as of the current build.

---

## Key Files

| File | Role | When it runs |
|------|------|-------------|
| `_scripts/inject_juv_metadata.py` | Inject PEP 723 cell | Once per notebook (manual / skill) |
| `_scripts/generate_bundles.py` | Create zips + manifest | `make build` |
| `_scripts/test_bundles.py` | Validate bundles | `make test-bundles` |
| `assets/download-bundle.js` | Inject buttons on pages | Reader's browser |
| `styles/_download-bundle.scss` | Button styling | Compiled by Quarto |
| `includes/download-bundle.html` | `<script>` tag include | Quarto `include-after-body` |

---

## Rebuilding from Scratch

If you lose the generated bundles or need to regenerate:

```bash
# 1. Re-inject PEP 723 cells into all notebooks (if lost)
for nb in posts/*/index.ipynb; do
  python3 _scripts/inject_juv_metadata.py "$nb"
done

# 2. Full rebuild (render + JupyterLite + LLM context + bundles)
make clean && make build

# 3. Verify
make test-bundles

# 4. Deploy
make deploy
```

The PEP 723 cells are committed to git, so step 1 is only needed if they've been accidentally removed from the notebooks themselves.
