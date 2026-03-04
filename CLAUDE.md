# Claude Session Context

## Project Overview
This is a Quarto-based personal website for data science/ML educational content at rahuldave.github.io.

## Content Source
- AM207 course wiki at `~/Attic/Projects/AM207/2018fall_wiki/wiki/` — lectures, notebooks, markdown notes
- Lecture index at `~/Attic/Projects/AM207/2018fall_wiki/lectures/index.md` (replace `.html` with `.md` for source files)
- Each lecture `.md` links to wiki notes; check for `.ipynb` (primary source) before falling back to `.md`
- Use `/import-wiki-notes` skill to import notes as blog posts
- **Helper scripts** in `_scripts/`:
  - `import_notebook.py` — creates `posts/<name>/index.ipynb` from wiki `.ipynb` (adds frontmatter, fixes paths, copies images). Run with `--help`.
  - `update_captions.py` — updates image captions in notebook markdown cells. Used by `/caption-images` skill.
  - `inject_juv_metadata.py` — scans notebooks for imports, injects PEP 723 dependency cell for `juv`. Used by `/bundle-post` skill.
  - `generate_bundles.py` — generates `<slug>.zip` bundles in `_site/` at build time. Run via `make bundles` or `make build`.
  - `test_bundles.py` — tests bundles by running `uvx juv exec`. Run via `make test-bundles`.
  - `execute_notebook.py` — executes a notebook in-place with `nbclient`, capturing outputs. Self-bootstraps PEP 723 deps via `uv run --with`. Usage: `uv run _scripts/execute_notebook.py [--timeout 1200] posts/SLUG/index.ipynb`. Used by `/execute-notebook` skill.
  - `run_nb.py` — quick-test: runs notebook cells via `exec()`, reports first failure. No output capture. Usage: `uv run --with <deps> python _scripts/run_nb.py posts/SLUG/index.ipynb`.
  - `generate_learning_paths.py` — generates per-path `.qmd` pages + `assets/learning-paths.json` manifest from YAML DAGs in `_learning-paths/`. Run via `make build`.
  - `build_jupyterlite.sh` — builds JupyterLite static site into `_site/lab/`. Auto-discovers labextension path. Run via `make build`.
  - `pymc3-to-pymc-porting.md` — comprehensive reference for migrating pymc3/theano notebooks to modern pymc/pytensor.
- After importing, run `/caption-images` to add captions, then `/bundle-post` for notebook posts

## Content Import Status (from AM207 wiki)
- **Lecture 1 (Intro & Probability)**: DONE
  - `boxloop` — already existed, skipped
  - `probability` — notebook, imported to `posts/probability/`
  - `distributions` — markdown only, imported to `posts/distributions.md`
  - `distrib-example` — notebook, imported to `posts/distrib-example/`
- **Lecture 2 (Probability, Sampling, Laws, Monte Carlo)**: DONE
  - `distributions` — already imported (Lecture 1), skipped
  - `expectations` — notebook, imported to `posts/expectations/`
  - `samplingclt` — notebook, imported to `posts/samplingclt/`
  - `basicmontecarlo` — notebook, imported to `posts/basicmontecarlo/`
  - `montecarlointegrals` — notebook, imported to `posts/montecarlointegrals/`
- **Lecture 3 (From Monte Carlo to Frequentist Stats)**: DONE
  - `Expectations` — already imported (Lecture 2), skipped
  - `SamplingCLT` — already imported (Lecture 2), skipped
  - `basicmontecarlo` — already imported (Lecture 2), skipped
  - `montecarlointegrals` — already imported (Lecture 2), skipped
  - `frequentist` — markdown only, imported to `posts/frequentist.md`
  - `MLE` — notebook, imported to `posts/MLE/`
- **Lecture 4 (MLE, Sampling, and Learning)**: DONE
  - `noiseless_learning` — notebook, imported to `posts/noiseless_learning/`
  - `noisylearning` — notebook, imported to `posts/noisylearning/`
  - `MLE` — already imported (Lecture 3), skipped
  - `testingtraining` — notebook, imported to `posts/testingtraining/`
  - `validation` — notebook, imported to `posts/validation/`
  - `regularization` — notebook, imported to `posts/regularization/`
- **Lecture 5 (Regression, AIC, Info. Theory)**: DONE
  - `doseplacebo` — notebook, imported to `posts/doseplacebo/`
  - `noiseless_learning` — already imported (Lecture 4), skipped
  - `noisylearning` — already imported (Lecture 4), skipped
  - `testingtraining` — already imported (Lecture 4), skipped
  - `validation` — already imported (Lecture 4), skipped
  - `regularization` — already imported (Lecture 4), skipped
  - `jensens` — notebook, imported to `posts/jensens/`
  - `Divergence` — notebook, imported to `posts/divergence/`
  - `understandingaic` — notebook, imported to `posts/understandingaic/`
- **Lecture 6 (Risk and Information)**: DONE
  - `jensens` — already imported (Lecture 5), skipped
  - `Divergence` — already imported (Lecture 5), skipped
  - `understandingaic` — already imported (Lecture 5), skipped
  - `Entropy` — notebook, imported to `posts/entropy/`
- **Lecture 7 (From Entropy to Bayes)**: DONE
  - `Divergence` — already imported (Lecture 5), skipped
  - `Entropy` — already imported (Lecture 6), skipped
  - `bayes_withsampling` — notebook, imported to `posts/bayes_withsampling/`
  - `globemodel` — notebook, imported to `posts/globemodel/`
  - `sufstatexch` — notebook, imported to `posts/sufstatexch/`
- **Lecture 8 (Bayes and Sampling)**: DONE
  - `bayes_withsampling` — already imported (Lecture 7), skipped
  - `globemodel` — already imported (Lecture 7), skipped
  - `sufstatexch` — already imported (Lecture 7), skipped
  - `globemodellab` — notebook, imported to `posts/globemodellab/`
  - `normalmodel` — notebook, imported to `posts/normalmodel/`
  - `lightspeed` — source not found in wiki, skipped
  - `inversetransform` — notebook, imported to `posts/inversetransform/`
  - `rejectionsampling` — notebook, imported to `posts/rejectionsampling/`
  - `importancesampling` — notebook, imported to `posts/importancesampling/`
- **Lecture 9 (Bayes and Sampling, cont.)**: DONE
  - `bayes_withsampling` — already imported (Lecture 7), skipped
  - `normalmodel` — already imported (Lecture 8), skipped
  - `sufstatexch` — already imported (Lecture 7), skipped
  - `lightspeed` — source not found in wiki, skipped
  - `inversetransform` — already imported (Lecture 8), skipped
  - `rejectionsampling` — already imported (Lecture 8), skipped
  - `bayesianregression` — notebook, imported to `posts/bayesianregression/`
  - `normalreg` — notebook, imported to `posts/normalreg/` (includes Howell1.csv data)
- **Lecture 10 (Sampling and Gradient Descent)**: DONE
  - `inversetransform` — already imported (Lecture 8), skipped
  - `rejectionsampling` — already imported (Lecture 8), skipped
  - `importancesampling` — already imported (Lecture 8), skipped
  - `gradientdescent` — notebook, imported to `posts/gradientdescent/` (SGD gif replaced with link)
  - `LogisticBP` — notebook, imported to `posts/logisticbp/`
- **Lecture 11 (Gradient Descent and Neural Networks)**: DONE
  - `gradientdescent` — already imported (Lecture 10), skipped
  - `LogisticBP` — already imported (Lecture 10), skipped
  - `FuncTorch` — notebook, imported to `posts/functorch/`
  - `nnreg` — notebook, imported to `posts/nnreg/`
  - `MLP_Classification` — notebook, imported to `posts/mlp_classification/`
- **Lecture 12 (Non Linear Approximation to Classification)**: DONE
  - `LogisticBP` — already imported (Lecture 10), skipped
  - `FuncTorch` — already imported (Lecture 11), skipped
  - `nnreg` — already imported (Lecture 11), skipped
  - `MLP_Classification` — already imported (Lecture 11), skipped
  - `typesoflearning` — notebook, imported to `posts/typesoflearning/`
  - `generativemodels` — notebook, imported to `posts/generativemodels/` (includes heights/weights CSV)
  - `utilityorrisk` — notebook, imported to `posts/utilityorrisk/`
- **Lecture 13 (Classification, Mixtures, and EM)**: DONE
  - `typesoflearning` — already imported (Lecture 12), skipped
  - `generativemodels` — already imported (Lecture 12), skipped
  - `EM` — notebook, imported to `posts/em/`
- **Lecture 14 (EM and Hierarchical Models)**: DONE
  - `EM` — already imported (Lecture 13), skipped
  - `hierarch` — notebook, imported to `posts/hierarch/`
- **Lecture 15 (MCMC)**: DONE
  - `markov` — notebook, imported to `posts/markov/`
  - `metropolis` — notebook, imported to `posts/metropolis/`
  - `discretemcmc` — notebook, imported to `posts/discretemcmc/`
  - `bayes_withsampling` — already imported (Lecture 7), skipped
  - `introgibbs` — notebook, imported to `posts/introgibbs/`
  - `hierarch` — already imported (Lecture 14), skipped
- **Lecture 16 (MCMC and Gibbs)**: DONE
  - `markov` — already imported (Lecture 15), skipped
  - `metropolis` — already imported (Lecture 15), skipped
  - `discretemcmc` — already imported (Lecture 15), skipped
  - `bayes_withsampling` — already imported (Lecture 7), skipped
  - `introgibbs` — already imported (Lecture 15), skipped
  - `hierarch` — already imported (Lecture 14), skipped
  - `metropolishastings` — notebook, imported to `posts/metropolishastings/`
  - `MetropolisSupport` — notebook, imported to `posts/metropolissupport/`
  - `switchpoint` — notebook, imported to `posts/switchpoint/`
  - `gibbsfromMH` — markdown, imported to `posts/gibbsfromMH/`
  - `gibbsconj` — notebook, imported to `posts/gibbsconj/`
- **Lecture 17 (Data Augmentation, Gibbs, and HMC)**: DONE
  - `dataaug` — notebook, imported to `posts/dataaug/`
  - `tetchygibbs` — notebook, imported to `posts/tetchygibbs/`
  - `hmcidea` — markdown, imported to `posts/hmcidea/`
  - `hmcexplore` — notebook, imported to `posts/hmcexplore/`
- **Lectures 18–26**: NOT YET IMPORTED

## PyMC3 → PyMC Migration Status
- **Notebooks already ported**: `em` (removed unused import), `hmcexplore` (removed unused import), `switchpoint` (full InferenceData port), `utilityorrisk` (full InferenceData port)
- **Porting reference**: `_scripts/pymc3-to-pymc-porting.md`
- **Skill**: `/port-pymc3` — step-by-step process for migrating pymc3/theano to modern pymc/pytensor
- **Critical**: Do NOT list `arviz` explicitly in PEP 723 deps alongside `pymc` — arviz 1.0 breaks pymc; let pymc pull in the compatible version
- **Known slow notebooks** (use `--timeout 1200`): switchpoint (~90s), utilityorrisk (~70s), mlp_classification (~90s), nnreg (~80s), samplingclt (~230s), gibbsconj (~50s), tetchygibbs (~40s)
- **All 49 notebook bundles pass** `make test-bundles` (100%)

## Modernization Debt (deprecated patterns still in notebooks)

### 1. `sns.distplot` → `sns.histplot`/`sns.displot` (seaborn, removed in v0.14)
5 notebooks: `bayes_withsampling`, `globemodellab`, `metropolissupport`, `normalreg`, `typesoflearning`

### 2. `switchpoint` pymc introspection shims
6 cells wrapped in try/except because pymc v5 changed internal APIs (transformed variable access, distribution internals, model logp). These work but print fallback messages instead of showing the original pedagogical content. Could be rewritten to use modern pymc equivalents.

### 3. Stale stderr in cell outputs (cosmetic)
6 notebooks have old `axes.color_cycle` deprecation warnings baked into stderr from the original Python 3.5 environment: `divergence`, `expectations`, `jensens`, `markov`, `montecarlointegrals`, `noisylearning`. The code is fine — these are just leftover output strings from cells with `%matplotlib inline` that `execute_notebook.py` skips (it doesn't execute magic lines). Re-running these cells manually or stripping stderr would clean them up.

### 4. Stale kernel metadata (cosmetic)
7 notebooks still show old `kernelspec.display_name` (e.g. "Python [conda env:py35]", "Python [default]"): `basicmontecarlo`, `distrib-example`, `expectations`, `montecarlointegrals`, `probability`, `samplingclt`, `votingforcongress`. The code runs on Python 3.14 — only the metadata string is stale. `nbclient` preserves existing kernelspec metadata.

## Post Date Scheme
- Dates increase by **1 week per lecture**, starting from 2025-01-08 (Lecture 1)
- All notes within a lecture share the same date (the lecture's date)
- If a note was first imported in an earlier lecture, it keeps that earlier date
- **Last date used**: 2025-04-30 (Lecture 17)
- **Next lecture (18) should use**: 2025-05-07

## Category System
- Canonical categories are in `_categories.txt` (root of project), one per line, sorted alphabetically
- **Current categories**: bayesian, classification, data, decision-theory, elections, hierarchical, information-theory, integration, interactive, macos, mcmc, models, montecarlo, neural-networks, optimization, orchestration, pipeline, probability, regression, sampling, statistics, visualization
- All categories must be lowercase
- When importing, map source keywords to existing categories; propose new ones for user approval
- The `/import-wiki-notes` skill enforces this workflow (step 7c)

## About Page
- Profile photo: `assets/profile.jpg` (sourced from GitHub avatar)
- Comprehensive bio context: `~/Context/rahul-dave-profile.md`
- Contact form via Formspree (ID: `mykjwlyk`) — forwards to rahuldave@univ.ai
- Links: GitHub, X/Twitter, Bluesky (rahuldave.bsky.social), Google Scholar, LinkedIn

## Social / Discussion Links
- `includes/discuss-links.html` — "Discuss this post" links (Twitter, Bluesky, LinkedIn) on posts, til, and collections pages only
- Uses IIFE (not DOMContentLoaded) because `include-after-body` runs after DOM is ready
- Links are constructed at runtime from `window.location.href` — will use production domain automatically
- Open Graph and Twitter Card metadata enabled in `_quarto.yml`

## Listing Sort Order
- All listing pages sort by `date desc`: `index.qmd`, `posts.qmd`, `til.qmd`, `collections.qmd`
- `learning-paths.qmd` sorts by `order` (beginner=1, intermediate=2, advanced=3)

## Design
- `designs/design1-depth/` — **CHOSEN BASE DESIGN** (Blues sequential palette, clean, scholarly)
- `designs/design1-modern/` — fork of depth, modernized with animations/glassmorphism
- Serif typography: Bitter (headings) + Source Serif 4 (body) + IBM Plex Mono (code)
- ColorBrewer Blues palette (#eff3ff → #08306b)
- Dark/light mode support

### ColorBrewer Blues Palette
```
--cb-blue-50: #eff3ff   --cb-blue-100: #c6dbef   --cb-blue-200: #9ecae1
--cb-blue-300: #6baed6  --cb-blue-400: #4292c6   --cb-blue-500: #2171b5
--cb-blue-600: #08519c  --cb-blue-700: #084594   --cb-blue-800: #08306b
```

---

## Quarto Site Structure

### Config
- `_quarto.yml` — project config: website type, navbar, SCSS theme (light/dark), TOC, margin references, Open Graph, Twitter Cards
- Themes: `styles/modern-light.scss`, `styles/modern-dark.scss`
- Includes: `includes/fonts.html`, `includes/brand-icon.html`, `includes/theme-toggle.html`, `includes/discuss-links.html`, `includes/learning-paths.html`

### Posts Directory Layout

**Notebooks** get their own folder with `index.ipynb` (or `index.qmd` for JS demos):
```
posts/
  probability/
    index.ipynb        # URL becomes /posts/probability/
    assets/            # images, data files referenced by THIS notebook only
      venn.png
      bishop-prob.png
  votingforcongress/
    index.ipynb
    assets/
      sep7.png
  earth-demo/
    index.qmd          # Three.js demo using .qmd format
    assets/
      earth-card.png   # Card thumbnail for listing (no content images in post)
```

**Markdown posts** stay as flat files, images go in shared `posts/images/` or `posts/data/`:
```
posts/
  boxloop.md           # URL becomes /posts/boxloop
  distributions.md
  images/              # shared images for flat .md posts
    2tosscdf.png
  data/                # shared data for flat .md posts (if needed)
```

### Notebook Frontmatter (Quarto YAML)
Notebooks must have a **raw cell** (cell_type: "raw") as the first cell with YAML:
```yaml
---
title: "Post Title"
subtitle: "Catchy one-liner for listing cards."
description: "Two-sentence summary for the post page."
categories:
    - probability
    - statistics
date: 2025-01-08
---
```
- `subtitle` appears on listing cards
- `description` is the longer summary
- `categories` are used for filtering (must be from `_categories.txt`)
- `date` controls sort order
- `image` — optional, for posts without content images (e.g. `image: assets/earth-card.png`)

For markdown posts, the same YAML goes in the standard frontmatter block at the top.

### Card Images for Posts Without Content Images
For posts with no embedded images (e.g. interactive Three.js demos), use the browser agent to screenshot the rendered page, crop to the key visual, save to `assets/`, and add `image:` to frontmatter. See skill step 9 for details.

### Three.js / Interactive .qmd Posts
Use `.qmd` format (not `.ipynb`) for JavaScript-heavy interactive content:
- Load external JS via `format: html: include-in-header:` in frontmatter
- Use Quarto's fenced div syntax `::: {.classname}` for layout containers
- Inline `<script>` blocks at the end of the .qmd file
- Read CSS custom properties (`--cb-blue-*`, `--interactive-*`) for theme integration
- Example: `posts/earth-demo/index.qmd`

### Build & Deploy (Makefile)
```bash
make preview                # Live dev server with hot reload
make build                  # Compile prompts + render + JupyterLite + LLM context + zip bundles
make deploy                 # Build + push _site/ to gh-pages branch
make test-bundles           # Test bundles via juv exec
make clean                  # Delete stamp files (forces full rebuild)
quarto render posts/probability/index.ipynb  # Render a single post
```
- `build` runs: `compile_prompts.py` + `generate_learning_paths.py` → `quarto render` → `build_jupyterlite.sh` + `generate_llm_context.py` → `generate_bundles.py` (skips stages whose inputs haven't changed, using stamp files)
- `deploy` uses `git worktree` to check out `gh-pages` into `/tmp/`, rsync `_site/` there, commit, push
- GitHub Pages deploys from `gh-pages` branch (root `/`), NOT from `docs/` on `main`
- `CNAME` and `.nojekyll` live in project root — Quarto copies them to `_site/` automatically
- Commit and push of `main` are separate from deploy
- **Full internal docs:** `_internal_docs/build-and-deploy.md`
- **IMPORTANT: ALWAYS use Makefile targets** (`make build`, `make deploy`, `make clean`, etc.) instead of running scripts or `quarto render` directly. The Makefile orchestrates the correct build order (render → JupyterLite → LLM context → bundles) with stamp-based incremental builds. Running `quarto render` alone blows away `_site/` and loses JupyterLite, bundles, and LLM context artifacts. The only exception is rendering a single post for quick iteration: `quarto render posts/SLUG/index.ipynb`.

### Image/Data Path Conventions
- Notebook posts: `assets/filename.png` (relative to the notebook's folder)
- Markdown posts: `images/filename.png` (relative to posts/)
- Site-wide assets already in `/assets/` (e.g. lawoflargenumbers images) use `/assets/...` absolute paths
- Profile photo: `assets/profile.jpg` (JPEG, not PNG — watch for GitHub avatar format mismatch)

### Categories
- **All categories must be lowercase** — no `Statistics`, use `statistics`; no `MonteCarlo`, use `montecarlo`
- Canonical list in `_categories.txt` at project root
- This prevents duplicate tags in Quarto listing filters

### Frontmatter Fields
- `title` — post title
- `subtitle` — catchy one-liner shown on listing cards
- `description` — 2-sentence content summary
- `categories` — lowercase tags for filtering (from `_categories.txt`)
- `date` — controls sort order (YYYY-MM-DD)
- `image` — card thumbnail path (optional, for posts without content images)

### Site Sections
- `posts/` — main blog posts (shown on index page)
- `til/` — Today I Learned (shown only on TIL page, NOT on index)
- `collections/software/` — software tools (shown only on Collections page, NOT on index)
- `_lab/` — JupyterLite source config (underscore prefix = Quarto ignores it; built to `_site/lab/`)
- `_learning-paths/` — YAML DAG definitions + optional content/images for learning paths (underscore prefix = Quarto ignores)
- `learning-paths/` — generated `.qmd` pages (gitignored, rebuilt by `make build`)
- `index.qmd` listing contents should only include `posts/` and `posts/**/`

### LLM Explain Feature (BYOK)
Readers use their own Claude API key (browser localStorage) to get AI explanations. No backend — Anthropic API supports CORS directly via `anthropic-dangerous-direct-browser-access: true` header.

**Architecture (3 layers):**
1. **Lua filter** (`_filters/cell-markers.lua`) — adds `data-cell-type="code"` to notebook cell divs
2. **Python script** (`_scripts/generate_llm_context.py`) — generates `_content.md` + `cells.json` per post in `_site/`
3. **Runtime JS** (`assets/llm-explain.js`) — button injection, API key modal, streaming chat modal with markdown+MathJax rendering

**Config is externalized** to `_llm-config.yml` (model, max_tokens, prompts) → compiled by `_scripts/compile_prompts.py` → `assets/llm-prompts.json` (committed, copied to `_site/` by Quarto). The JS fetches the JSON at runtime with hardcoded fallbacks. Edit `_llm-config.yml` and run `python3 _scripts/compile_prompts.py` to update (also runs automatically via `make build`).

**Key files:** `assets/llm-explain.js`, `styles/_llm-explain.scss`, `_filters/cell-markers.lua`, `_scripts/generate_llm_context.py`, `_llm-config.yml`, `_scripts/compile_prompts.py`, `assets/llm-prompts.json`, `includes/llm-explain.html` (just a `<script src>` tag)

**LLM context files use `_content.md`** (underscore prefix) because Quarto skips `_`-prefixed files. Without the underscore, Quarto would scan them as renderable source and break listings. `cells.json` is fine (`.json` isn't a Quarto source type).

**Full internal docs:** `_internal_docs/llm-explain-system.md`

### Download & Run Bundles (juv + zip)
Each notebook post gets a downloadable `<slug>.zip` containing:
- `index.ipynb` — source notebook with PEP 723 inline dependencies (`#| include: false` hides the cell in Quarto)
- `assets/` — images (PNG, JPG, etc.) and per-notebook data files (CSV)
- `data/` — data files if present (CSV, etc.)
- `README.md` — title, link to online version, quick-start instructions, dependency list

Readers unzip and run with `uvx juv run index.ipynb`. The PEP 723 cell tells `juv` which packages to install.

**Architecture:**
1. **PEP 723 injection** (`_scripts/inject_juv_metadata.py`) — scans imports, injects hidden dependency cell at index 1 with `#| include: false`
2. **Bundle generation** (`_scripts/generate_bundles.py`) — zips notebook + assets/ + data/ + README.md into `_site/posts/<slug>/<slug>.zip`, writes `_site/bundles.json` manifest
3. **Runtime JS** (`assets/download-bundle.js`) — HEAD-checks for zip, injects "Download & Run" button on post pages

**Key files:** `assets/download-bundle.js`, `styles/_download-bundle.scss`, `includes/download-bundle.html`, `_scripts/inject_juv_metadata.py`, `_scripts/generate_bundles.py`, `_scripts/test_bundles.py`

**Workflow for new notebook posts:** Run `/bundle-post` (called automatically by `/finalize-post`). This checks for missing data files, injects PEP 723 deps, and verifies bundle contents. The zip is generated at build time by `make build`.

**Full internal docs:** `_internal_docs/download-and-run.md`

### Run in Browser (JupyterLite)
Pyodide-compatible notebooks (43 of 49) can run entirely in the reader's browser via JupyterLite. A "Run in Browser" button on each compatible post links to `/lab/loader.html?zip=...` which fetches the zip bundle, rewrites PEP 723 deps for micropip, writes files to JupyterLite's IndexedDB, and redirects to the JupyterLite Lab interface.

**Key files:** `_lab/jupyter-lite.json` (preloaded packages config), `_lab/loader.html` (zip→IndexedDB bridge), `_scripts/build_jupyterlite.sh` (builds JupyterLite static site), `assets/download-bundle.js` (injects both buttons), `styles/_download-bundle.scss`

**Preloaded Pyodide built-ins (7):** numpy, scipy, matplotlib, pandas, scikit-learn, statsmodels, Pillow. Pure Python packages (e.g. seaborn) are installed at runtime via `micropip.install()`.

**Full internal docs:** `_internal_docs/run-in-browser.md`

### Learning Paths (Guided Sequences)
Structured sequences of posts organized into named parts, defined as YAML DAGs. Each path gets a listing-page card, a detail page with step cards, and JavaScript-powered prev/next navigation on post pages when a reader follows a path.

**Architecture (3 layers):**
1. **Build-time Python** (`_scripts/generate_learning_paths.py`) — reads `_learning-paths/*.yml`, topologically sorts nodes (Kahn's algorithm), generates per-path `.qmd` pages + `assets/learning-paths.json` manifest
2. **Runtime JS** (`assets/learning-paths.js`) — IIFE on every post page; checks for `?path=<id>&step=<n>` query param; injects top banner ("Part N: Title — Step X of Y") and bottom prev/next nav cards
3. **SCSS styling** (`styles/_learning-paths.scss`) — banner, nav cards, step cards on path pages

**To add a new learning path:** Create `_learning-paths/<id>.yml` (YAML DAG with parts/nodes), optionally add `<id>-content.md` (intro prose) and `<id>-card.png` (listing thumbnail from a post diagram), then `make build`.

**Key files:** `_learning-paths/*.yml`, `_scripts/generate_learning_paths.py`, `assets/learning-paths.js`, `styles/_learning-paths.scss`, `includes/learning-paths.html`, `learning-paths.qmd`

**Generated (gitignored):** `learning-paths/*.qmd`, `assets/learning-paths.json`

**Full internal docs:** `_internal_docs/learning-paths.md`

### Notebook Execution (Refresh Outputs)
Quarto renders stored cell outputs, so notebooks must be executed before deploy for fresh results. The `/finalize-post` skill includes this as step 6.

```bash
uv run _scripts/execute_notebook.py posts/SLUG/index.ipynb           # default 600s timeout
uv run _scripts/execute_notebook.py --timeout 1200 posts/SLUG/index.ipynb  # slow notebooks
uv run _scripts/execute_notebook.py --allow-errors posts/SLUG/index.ipynb  # continue on errors
```

The script is self-bootstrapping: reads each notebook's PEP 723 deps and re-invokes itself via `uv run --with <deps>`. No pre-built environment needed.

For quick pass/fail testing (no output capture): `uv run --with <deps> python _scripts/run_nb.py posts/SLUG/index.ipynb`

**No `ipynb: default` in frontmatter.** The old "Other Formats > Jupyter" download is replaced by zip bundles. The `import_notebook.py` script and all skills omit `ipynb: default`.

**Makefile uses stamp files** (`_site/.stamp.*`) so `make deploy` after `make build` skips the render. Run `make clean` to force a full rebuild.

### Visual Testing with Browser Agent
- **ALWAYS use `npx agent-browser`** to visually verify changes to the site — don't just assume CSS/JS/layout changes look right. Spin up a local server (`python3 -m http.server 8765 --directory _site`) and use the browser agent to take screenshots, check responsiveness at different viewports (`agent-browser set viewport 375 812` for mobile), click interactive elements, and inspect console errors.
- Use the browser agent to test JupyterLite integration: navigate to the loader page, verify notebooks open, check IndexedDB contents, run cells, and confirm plots render.
- Use `agent-browser eval "..."` to inspect DOM state, IndexedDB, console output, and other runtime details that can't be checked by reading source code alone.
- For deployed site testing, navigate to `https://rahuldave.com/...`. For local testing, use `http://localhost:8765/...` with a local server serving `_site/`.
- Check both desktop (1280x800) and mobile (375x812) viewports when making layout/CSS changes.

### Known Gotchas
- After adding/moving/renaming files, restart `quarto preview` — the live server caches resource IDs and will show "Bad resource ID" for changed files
- Listing `.qmd` files must not have a stray `---` after the YAML block (causes parse errors)
- Listing `contents` paths should NOT have a leading `/` (use `til/*.qmd` not `/til/*.qmd`)
- `include-after-body` scripts run after DOMContentLoaded — use IIFE, not event listeners
- GitHub avatar downloads are JPEG even with `.png` URL — always check with `file` command and use correct extension
