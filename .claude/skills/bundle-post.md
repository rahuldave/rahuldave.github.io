# Skill: Bundle a Notebook Post

Prepare a notebook post for download-and-run by injecting PEP 723 dependency metadata and verifying all data files are present. The zip bundle itself is generated at build time by `make bundles`.

## Invocation

```
/bundle-post <post_path>
```

- `post_path`: path to the notebook post (e.g. `posts/probability/index.ipynb`)

**Only applies to notebook posts** (`posts/*/index.ipynb`). Skip for markdown posts, QMD posts, or notebooks with no Python code cells.

## Procedure

### 1. Check for Data Files

Scan all code cells for file-loading patterns:
- `read_csv('...')` / `read_csv("...")`
- `loadtxt('...')` / `genfromtxt('...')`
- `open('...')` with data file extensions
- `np.load('...')` / `pickle.load(...)`

For each referenced file path (typically `data/foo.csv` or `assets/bar.csv`):
1. Check if the file exists in the post directory
2. If missing, check the wiki data folder (`~/Attic/Projects/AM207/2018fall_wiki/wiki/data/`) and the wiki source directory for a match
3. Copy any found files into the appropriate subdirectory (`data/` or `assets/`)
4. If a referenced file cannot be found anywhere, warn the user

### 2. Inject PEP 723 Dependencies

Run the injection script on the single notebook:

```bash
python3 _scripts/inject_juv_metadata.py --posts-dir posts
```

This script is idempotent — it removes any existing PEP 723 cell and re-injects with current dependencies. It also ensures `ipynb: default` is not in the frontmatter.

Verify the injected cell by checking cell index 1:
- Should be a code cell with `# /// script` metadata
- Should have `"jupyter": {"source_hidden": true}` in metadata
- Dependencies should match the imports found in the notebook

### 3. Verify Bundle Contents

Check that everything needed for a self-contained bundle exists:

```bash
python3 -c "
import json, os, re
nb_path = '<post_path>'
post_dir = os.path.dirname(nb_path)
with open(nb_path) as f:
    nb = json.load(f)
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    # Check data file references
    for pattern in [r'read_csv\([\"\\']([^\"\\']*)[\"\\']\)', r'loadtxt\([\"\\']([^\"\\']*)[\"\\']\)']:
        for m in re.finditer(pattern, src):
            path = os.path.join(post_dir, m.group(1))
            status = 'OK' if os.path.exists(path) else 'MISSING'
            print(f'  {status}: {m.group(1)}')
    # Check image references
for c in nb['cells']:
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c['source'])
    for m in re.finditer(r'!\[.*?\]\((?!data:)(.*?)\)', src):
        path = os.path.join(post_dir, m.group(1))
        status = 'OK' if os.path.exists(path) else 'MISSING'
        print(f'  {status}: {m.group(1)}')
"
```

### 4. Report

Print a summary:
- Notebook slug
- Number of dependencies detected
- Pyodide compatibility (no torch/pymc3)
- Data files: present / missing
- Image files: present / missing

The actual zip bundle is generated at build time by `_scripts/generate_bundles.py` (run via `make bundles` or `make build`). This skill just ensures the notebook is ready for bundling.

## Notes

- The PEP 723 cell uses `source_hidden: true` so JupyterLab hides it from view
- The injection script is idempotent — safe to run multiple times
- Dependencies are detected from `import` statements only; if a notebook uses `np.` without `import numpy`, it won't be detected (this is a pre-existing issue in the notebook)
- Pyodide-incompatible packages: `torch`, `pymc3`, `theano-pymc`
- The zip bundle replaces the old `ipynb: default` "Other Formats > Jupyter" download
