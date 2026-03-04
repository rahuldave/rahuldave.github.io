# Run in Browser (JupyterLite) — Internal Documentation

## Overview

Pyodide-compatible notebooks can be executed entirely in the reader's browser via JupyterLite — no server, no local install. The system reuses the same zip bundles built for "Download and Run" but bridges them into JupyterLite through a loader page that rewrites dependencies for the Pyodide environment.

**User-facing flow:** Click "Run in Browser" (opens in new tab) → loader page fetches zip, unpacks it, writes to IndexedDB → JupyterLite opens with notebook ready to execute. Mobile users get the single-document Notebook interface; desktop users get the full JupyterLab IDE.

**Build-time flow:** Build JupyterLite static site → copy loader.html → generate bundles with `pyodide_compatible` flags.

---

## Architecture

```
User clicks "Run in Browser" on /posts/markov/
    │
    ▼
/lab/loader.html?zip=/posts/markov/markov.zip
    │
    │  1. Fetch zip
    │  2. Unzip with JSZip
    │  3. Parse notebook JSON
    │  4. Find PEP 723 cell, extract deps
    │  5. Subtract preloaded packages (numpy, scipy, etc.)
    │  6. Rewrite PEP 723 cell → micropip.install([extras]) or no-op
    │  7. Write all files to IndexedDB
    │  8. Redirect (mobile → Notebook, desktop → Lab)
    ▼
/lab/notebooks/index.html?path=markov/index.ipynb  (mobile, < 768px)
/lab/lab/index.html?path=markov/index.ipynb        (desktop)
    │
    │  JupyterLite loads notebook from IndexedDB
    │  Pyodide kernel starts with preloaded packages
    │  User runs cells (micropip cell installs extras first)
    ▼
Notebook executing in browser
```

---

## Build-Time Components

### 1. JupyterLite Static Site (`_scripts/build_jupyterlite.sh`)

Builds the JupyterLite application into `_site/lab/`. Run by `make build` via the `$(JLITE_STAMP)` Makefile target.

**Dependencies (installed system-wide):**
```bash
uv pip install --system --break-system-packages jupyterlite-core jupyterlite-pyodide-kernel
```

**Build command:**
```bash
jupyter lite build --lite-dir _lab --output-dir _site/lab \
    --FederatedExtensionAddon.extra_labextensions_path="$LABEXT_PATH"
```

- `--lite-dir _lab` — reads config from `_lab/jupyter-lite.json`
- `--output-dir _site/lab` — outputs into the site tree
- `--FederatedExtensionAddon.extra_labextensions_path` — required because Homebrew installs the Pyodide kernel extension at `/opt/homebrew/share/jupyter/labextensions/` which isn't in the default Jupyter data path

**Extension discovery:** The script auto-discovers the labextension path by importing `jupyterlite_pyodide_kernel` and checking candidate paths. Without this, the built JupyterLite has no kernel (`federated_extensions: []`) and shows "No Kernel" in the UI.

**Output structure:**
```
_site/lab/
├── lab/
│   └── index.html          ← JupyterLab IDE (desktop)
├── notebooks/
│   └── index.html          ← Notebook interface (mobile) — single-document, minimal toolbar
├── tree/
│   └── index.html          ← File browser (not directly used)
├── repl/
│   └── index.html          ← REPL console (not used — can't open .ipynb)
├── build/                   ← JS/CSS bundles (~62MB)
├── loader.html              ← Our custom loader (copied after build)
├── jupyter-lite.json        ← Compiled config
└── ...
```

Note the double `lab/` — JupyterLite outputs a `lab/` subdirectory inside our `_site/lab/` output directory. The Lab interface is at `/lab/lab/index.html`, the Notebook interface at `/lab/notebooks/index.html`.

### 2. JupyterLite Config (`_lab/jupyter-lite.json`)

Configures preloaded packages — Pyodide built-ins that load during kernel bootstrap:

```json
{
  "jupyter-lite-schema-version": 0,
  "jupyter-config-data": {
    "litePluginSettings": {
      "@jupyterlite/pyodide-kernel-extension:kernel": {
        "loadPyodideOptions": {
          "packages": [
            "numpy", "scipy", "matplotlib", "pandas",
            "scikit-learn", "statsmodels", "Pillow"
          ]
        }
      }
    }
  }
}
```

**Only Pyodide built-ins can be preloaded here.** These are packages compiled to WebAssembly and distributed with Pyodide. Pure Python packages like `seaborn` must be installed at runtime via `micropip.install()`.

**Current preloaded set (7 packages):** numpy, scipy, matplotlib, pandas, scikit-learn, statsmodels, Pillow.

**Why not seaborn?** Seaborn is pure Python, not a Pyodide built-in. Listing it in `loadPyodideOptions.packages` silently fails — it doesn't error but also doesn't install. It must go through micropip.

### 3. Loader Page (`_lab/loader.html`)

The bridge between zip bundles and JupyterLite. This is a standalone HTML page (not Quarto-rendered) that lives at `/lab/loader.html`.

**After the JupyterLite build, the Makefile copies it:**
```makefile
$(JLITE_STAMP): $(RENDER_STAMP) _lab/jupyter-lite.json _lab/loader.html _scripts/build_jupyterlite.sh
	_scripts/build_jupyterlite.sh
	cp _lab/loader.html _site/lab/loader.html
	@touch $@
```

### 4. Pyodide Compatibility Flags (`_scripts/generate_bundles.py`)

The `pyodide_incompatible` set determines which notebooks get a "Run in Browser" button:

```python
pyodide_incompatible = {"torch", "pymc3", "theano-pymc", "theano", "pymc", "lxml"}
```

This is checked against each notebook's PEP 723 dependencies. If any dep matches → `pyodide_compatible: false` in `bundles.json` → no button.

**Current status:** 43 compatible, 6 incompatible (torch: 3, pymc: 2, lxml: 1).

---

## Runtime Components

### Loader Logic (`_lab/loader.html`)

Invoked via: `/lab/loader.html?zip=/posts/<slug>/<slug>.zip`

**Step-by-step:**

1. **Parse query params** — extract `?zip=` URL and derive slug from `/posts/<slug>/`

2. **Fetch zip** — `fetch(zipUrl)` → `ArrayBuffer`

3. **Unzip** — JSZip (loaded from CDN: `jszip@3.10.1`)

4. **Find PEP 723 cell** — scans code cells for `# /// script` marker, extracts dependency list by parsing lines matching `#   "<pkg>",`

5. **Compute extra deps** — subtracts the `PRELOADED` set from PEP 723 deps:
   ```javascript
   var PRELOADED = new Set([
     "numpy", "scipy", "matplotlib", "pandas",
     "scikit-learn", "statsmodels", "pillow", "Pillow"
   ]);
   ```
   **This set must match `jupyter-lite.json` packages** (minus case variants). Package names are normalized (lowercased, dashes/underscores/dots collapsed) for comparison.

6. **Rewrite PEP 723 cell:**
   - If extras exist: → `import micropip\nawait micropip.install(["seaborn", ...])`
   - If no extras: → `# Dependencies are preloaded in the JupyterLite environment`
   - Clears cell outputs and metadata tags

7. **Write to IndexedDB** — populates JupyterLite's virtual filesystem (using the DB for the target app)

8. **Detect viewport** — `window.innerWidth < 768` → mobile (Notebook) or desktop (Lab)

9. **Redirect** — mobile: `/lab/notebooks/index.html?path=...`, desktop: `/lab/lab/index.html?path=...`

### Mobile-Adaptive Interface

The loader detects viewport width at the start and uses it for two decisions:

1. **Which IndexedDB to write to** — each JupyterLite app scopes its DB by base URL:
   - Desktop (Lab): `"JupyterLite Storage - /lab/"`
   - Mobile (Notebook): `"JupyterLite Storage - /lab/notebooks/"`
2. **Which interface to redirect to** — `/lab/lab/` (multi-pane IDE) vs `/lab/notebooks/` (single-document, one column of cells)

The Notebook interface is much better on mobile: no sidebar, no tab system, cells stack vertically.

### IndexedDB Schema

JupyterLite uses localforage to store files in IndexedDB.

**Database name:** `"JupyterLite Storage - /lab/"` for Lab, `"JupyterLite Storage - /lab/notebooks/"` for Notebook (scoped to each app's base path)

**Object store:** `"files"`

**Key format:** plain paths like `"markov/index.ipynb"` (no leading slash, no `/drive/` prefix)

**Three types of entries:**

#### Directory entries (required for file browser navigation)
```javascript
{
  name: "markov",
  path: "markov",
  format: "json",
  type: "directory",
  content: null,
  created: "2026-03-04T00:00:00.000Z",
  last_modified: "2026-03-04T00:00:00.000Z",
  writable: true,
  mimetype: "",
  size: 0
}
```

Without directory entries, the file browser sidebar is empty (even though files are accessible by direct path). The `filebrowser:go-to-path` command fails with "Item does not exist" because it can't navigate the directory tree.

The loader creates directory entries for the slug directory and all subdirectories (e.g., `markov`, `markov/assets`).

#### Notebook files (format: "json")
```javascript
{
  name: "index.ipynb",
  path: "markov/index.ipynb",
  format: "json",          // NOT "text" — must be "json"
  type: "file",
  content: { /* parsed notebook object */ },  // NOT a string
  created: "...",
  last_modified: "...",
  writable: true,
  mimetype: "application/x-ipynb+json",
  size: 87074
}
```

**Critical:** Notebooks must be stored as parsed JSON objects with `format: "json"`. Storing as a stringified JSON string with `format: "text"` causes JupyterLite to fail silently.

#### Text files (CSV, JSON, Python, etc.)
```javascript
{
  name: "data.csv",
  path: "markov/data/data.csv",
  format: "text",
  type: "file",
  content: "col1,col2\n1,2\n...",
  created: "...",
  last_modified: "...",
  writable: true,
  mimetype: "text/csv",
  size: 1234
}
```

#### Binary files (images, etc.)
```javascript
{
  name: "plot.png",
  path: "markov/assets/plot.png",
  format: "base64",
  type: "file",
  content: "iVBORw0KGgo...",
  created: "...",
  last_modified: "...",
  writable: true,
  mimetype: "image/png",
  size: 56789
}
```

### Button Injection (`assets/download-bundle.js`)

On each post page, the script fetches `bundles.json` and checks `pyodide_compatible`. If true, injects a "Run in Browser" button linking to `/lab/loader.html?zip=<zipPath>` with `target="_blank"` (opens in new tab).

Button is inserted as the leftmost item in the `.llm-summarize-wrap` bar:

`[▶ Run in Browser]` `[⤓ Download and Run · 164 KB]` `[Download md]` `[Summarize]`

Styled via `.run-in-browser-btn` in `styles/_download-bundle.scss` — blue accent variant of the base button style.

**Mobile layout:** On screens ≤ 768px, the `.llm-summarize-wrap` switches from flex-wrap to a 2×2 CSS grid layout (`grid-template-columns: 1fr 1fr`) so buttons align evenly instead of wrapping unevenly.

---

## Makefile Integration

```makefile
JLITE_STAMP := _site/.stamp.jupyterlite

$(JLITE_STAMP): $(RENDER_STAMP) _lab/jupyter-lite.json _lab/loader.html _scripts/build_jupyterlite.sh
	_scripts/build_jupyterlite.sh
	cp _lab/loader.html _site/lab/loader.html
	@touch $@
```

Position in the pipeline: after `quarto render` (needs `_site/` to exist), before bundles (which depend on it).

```
RENDER_STAMP → JLITE_STAMP ─┐
                             ├→ BUNDLES_STAMP
RENDER_STAMP → LLM_STAMP ───┘
```

---

## Key Files

| File | Role | When |
|------|------|------|
| `_lab/jupyter-lite.json` | Pyodide kernel config (preloaded packages) | Build time |
| `_lab/loader.html` | Zip → IndexedDB → JupyterLite bridge | Runtime (browser) |
| `_scripts/build_jupyterlite.sh` | Build JupyterLite static site | `make build` |
| `_scripts/generate_bundles.py` | Sets `pyodide_compatible` flag | `make build` |
| `assets/download-bundle.js` | Injects "Run in Browser" button | Runtime (browser) |
| `styles/_download-bundle.scss` | Button styling | Compiled by Quarto |
| `.gitignore` | Ignores `.jupyterlite.doit.db` | — |

---

## Gotchas & Lessons Learned

### 1. DB name is path-scoped
JupyterLite names its IndexedDB `"JupyterLite Storage - <baseUrl>"` — scoped to each app's base URL path. The Lab app uses `/lab/`, the Notebook app uses `/lab/notebooks/`. The loader must write to the correct DB for the target app. If the deployment path changes, update the DB name logic in `loader.html`.

### 2. Directory entries are required
Without explicit directory entries in IndexedDB, the file browser sidebar is empty. Files are still accessible by direct path, but the UI can't navigate to them. The `go-to-path` command fails.

### 3. Notebooks must be stored as parsed JSON
`format: "json"` + parsed object, NOT `format: "text"` + JSON string. JupyterLite silently fails with the wrong format.

### 4. Only Pyodide built-ins can be preloaded
`loadPyodideOptions.packages` only works for packages compiled into Pyodide (numpy, scipy, etc.). Pure Python packages (seaborn, etc.) must be installed via `micropip.install()` at runtime. Listing them in preloaded silently does nothing.

### 5. Extension discovery is platform-dependent
Homebrew puts labextensions at `/opt/homebrew/share/jupyter/labextensions/` which isn't in the default Jupyter data path. The build script auto-discovers this, but if it fails, JupyterLite builds with no kernel.

### 6. The double `/lab/lab/` path
JupyterLite outputs a `lab/` subdirectory inside the output directory. Since our output dir is `_site/lab/`, the Lab interface is at `_site/lab/lab/index.html` → URL `/lab/lab/index.html`. This is intentional.

### 7. `?path=` parameter is consumed and stripped
JupyterLite reads the `?path=` query parameter on load, opens the specified file, then strips the parameter from the URL bar. This is normal behavior — the path was received even though it disappears from the URL.

### 8. PRELOADED set must stay in sync
The `PRELOADED` set in `loader.html` must match the `packages` list in `jupyter-lite.json`. If they diverge, either micropip will try to install already-preloaded packages (wasted time) or skip packages that aren't actually available (import errors).

### 9. 404s for `all.json` are expected
JupyterLite tries to fetch pre-built content manifests (`/lab/api/contents/all.json`, etc.) on load. These don't exist because we populate files dynamically via IndexedDB. The console shows 404s with "don't worry" messages — this is normal.

---

## Rebuilding from Scratch

If you lose the JupyterLite build or loader:

```bash
# 1. Ensure dependencies are installed
uv pip install --system --break-system-packages jupyterlite-core jupyterlite-pyodide-kernel

# 2. Full rebuild
make clean && make build

# 3. Verify JupyterLite works
#    Start local server, open loader page in browser, check notebook loads
python3 -m http.server 8765 --directory _site
# Then visit: http://localhost:8765/lab/loader.html?zip=/posts/markov/markov.zip

# 4. Deploy
make deploy
```

If you need to change preloaded packages:
1. Edit `_lab/jupyter-lite.json` — add/remove from `packages` array
2. Edit `_lab/loader.html` — update the `PRELOADED` set to match
3. Run `make clean && make build` (JupyterLite must be rebuilt for config changes)

If you need to change the Pyodide incompatible set:
1. Edit `_scripts/generate_bundles.py` — update `pyodide_incompatible`
2. Run `make build` (regenerates `bundles.json` with updated flags)
