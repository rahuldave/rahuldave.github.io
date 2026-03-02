#!/usr/bin/env python3
"""Inject PEP 723 (juv) dependency metadata into notebook posts.

Scans each notebook's code cells for import statements, maps them to PyPI
packages, and injects a hidden PEP 723 metadata cell at index 1.
Also removes `ipynb: default` from frontmatter raw cells.

Usage:
    python3 _scripts/inject_juv_metadata.py [--posts-dir posts] [--dry-run]
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# ── Import name → PyPI package mapping ──────────────────────────────────
# Only entries where the import name differs from the PyPI package name,
# or where the import should be skipped entirely.

IMPORT_TO_PYPI = {
    # Different name on PyPI
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    # Same name (explicit for clarity)
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "torch": "torch",
    "pymc3": "pymc3",
    "statsmodels": "statsmodels",
    "tqdm": "tqdm",
    "IPython": "ipython",
    "theano": "theano-pymc",
}

# Modules to skip (stdlib, Jupyter built-ins, submodules handled via parent)
SKIP_MODULES = {
    # stdlib
    "abc", "argparse", "ast", "asyncio", "base64", "bisect", "builtins",
    "calendar", "cmath", "collections", "contextlib", "copy", "csv",
    "ctypes", "dataclasses", "datetime", "decimal", "difflib", "dis",
    "email", "enum", "errno", "fileinput", "fnmatch", "fractions",
    "functools", "gc", "getpass", "glob", "gzip", "hashlib", "heapq",
    "hmac", "html", "http", "importlib", "inspect", "io", "itertools",
    "json", "keyword", "linecache", "locale", "logging", "lzma", "math",
    "mimetypes", "multiprocessing", "numbers", "operator", "os", "pathlib",
    "pickle", "platform", "pprint", "profile", "pstats", "queue", "random",
    "re", "readline", "reprlib", "secrets", "select", "shelve", "shlex",
    "shutil", "signal", "site", "socket", "sqlite3", "ssl", "stat",
    "statistics", "string", "struct", "subprocess", "sys", "sysconfig",
    "tempfile", "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "traceback", "tracemalloc", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "venv", "warnings", "weakref", "webbrowser",
    "xml", "xmlrpc", "zipfile", "zipimport", "zlib",
    # Jupyter / IPython magics (not real imports)
    "__future__",
    # Sub-packages that come with a parent (don't add separately)
    "mpl_toolkits",  # comes with matplotlib
    "scipy.stats", "scipy.optimize", "scipy.integrate", "scipy.linalg",
    "scipy.special",  # comes with scipy
    "sklearn.linear_model", "sklearn.preprocessing", "sklearn.metrics",
    "sklearn.model_selection", "sklearn.datasets", "sklearn.lda",
    "sklearn.cross_validation", "sklearn.grid_search",  # comes with scikit-learn
    "torch.nn", "torch.autograd", "torch.utils",  # comes with torch
    "matplotlib.pyplot", "matplotlib.cm", "matplotlib.colors",
    "matplotlib.animation",  # comes with matplotlib
    "numpy.random", "numpy.linalg",  # comes with numpy
    "pandas.plotting",  # comes with pandas
    # Deprecated / unavailable
    "JSAnimation",  # old Jupyter animation, not on PyPI
}

# Packages that are NOT compatible with Pyodide
PYODIDE_INCOMPATIBLE = {"torch", "pymc3", "theano-pymc", "theano"}


def extract_imports(notebook: dict) -> set[str]:
    """Extract top-level module names from all code cells."""
    modules = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Skip cells that are already PEP 723 metadata
        if "# /// script" in source:
            continue
        for line in source.splitlines():
            line = line.strip()
            # Strip inline comments before parsing
            # (handles: import scipy as sp #imports stats functions, amongst other things)
            if "#" in line:
                line = line[:line.index("#")].strip()
            # Handle `import X`, `import X as Y`, `import X, Y`
            m = re.match(r"^import\s+(.+)", line)
            if m:
                for part in m.group(1).split(","):
                    mod = part.strip().split(" as ")[0].strip().split(".")[0]
                    if mod:
                        modules.add(mod)
            # Handle `from X import ...`
            m = re.match(r"^from\s+(\S+)\s+import", line)
            if m:
                mod = m.group(1).split(".")[0]
                if mod:
                    modules.add(mod)
    return modules


def resolve_dependencies(modules: set[str]) -> list[str]:
    """Map import names to sorted PyPI package names."""
    deps = set()
    for mod in modules:
        if mod in SKIP_MODULES:
            continue
        # Check mapping table
        if mod in IMPORT_TO_PYPI:
            deps.add(IMPORT_TO_PYPI[mod])
        else:
            # Assume PyPI name matches import name (lowercase)
            deps.add(mod.lower())
    return sorted(deps)


def is_pyodide_compatible(deps: list[str]) -> bool:
    """Check if all dependencies are Pyodide-compatible."""
    return not any(d in PYODIDE_INCOMPATIBLE for d in deps)


def build_pep723_cell(deps: list[str]) -> dict:
    """Build a PEP 723 inline script metadata cell for juv."""
    lines = [
        "#| include: false\n",
        "\n",
        "# /// script\n",
        '# requires-python = ">=3.10"\n',
        "# dependencies = [\n",
    ]
    for dep in deps:
        lines.append(f'#   "{dep}",\n')
    lines.append("# ]\n")
    lines.append("# ///\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "jupyter": {"source_hidden": True}
        },
        "outputs": [],
        "source": lines,
    }


def has_pep723_cell(notebook: dict) -> bool:
    """Check if notebook already has a PEP 723 metadata cell."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "# /// script" in source:
            return True
    return False


def remove_ipynb_format(notebook: dict) -> bool:
    """Remove `ipynb: default` from frontmatter raw cell. Returns True if changed."""
    if not notebook.get("cells"):
        return False
    cell = notebook["cells"][0]
    if cell.get("cell_type") != "raw":
        return False
    source = "".join(cell.get("source", []))
    if "ipynb: default" not in source and "ipynb:" not in source:
        return False

    # Remove the ipynb line(s) from the YAML block
    lines = cell.get("source", [])
    new_lines = []
    skip_next_indent = False
    for line in lines:
        stripped = line.strip()
        if stripped == "ipynb: default" or stripped == "ipynb:default":
            skip_next_indent = False
            continue
        if skip_next_indent and line.startswith("    "):
            continue
        skip_next_indent = False
        new_lines.append(line)

    if new_lines != lines:
        cell["source"] = new_lines
        return True
    return False


def process_notebook(nb_path: Path, dry_run: bool = False) -> dict | None:
    """Process a single notebook. Returns metadata dict or None if skipped."""
    with open(nb_path) as f:
        nb = json.load(f)

    # Skip if no code cells (markdown-only)
    has_code = any(c.get("cell_type") == "code" for c in nb.get("cells", []))
    if not has_code:
        return None

    slug = nb_path.parent.name
    modules = extract_imports(nb)
    deps = resolve_dependencies(modules)

    if not deps:
        return None

    already_has = has_pep723_cell(nb)
    changed = False

    # Remove existing PEP 723 cell if present (to re-inject with updated deps)
    if already_has:
        nb["cells"] = [
            c for c in nb["cells"]
            if not (c.get("cell_type") == "code" and "# /// script" in "".join(c.get("source", [])))
        ]

    # Inject PEP 723 cell at index 1 (after frontmatter raw cell)
    pep723_cell = build_pep723_cell(deps)
    nb["cells"].insert(1, pep723_cell)
    changed = True

    # Remove ipynb: default from frontmatter
    if remove_ipynb_format(nb):
        changed = True

    if changed and not dry_run:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return {
        "slug": slug,
        "dependencies": deps,
        "pyodide_compatible": is_pyodide_compatible(deps),
        "modules_found": sorted(modules),
        "already_had_pep723": already_has,
        "changed": changed,
    }


def main():
    parser = argparse.ArgumentParser(description="Inject PEP 723 metadata into notebooks")
    parser.add_argument("--posts-dir", default="posts", help="Posts directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    posts_dir = Path(args.posts_dir)
    notebooks = sorted(posts_dir.glob("*/index.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {posts_dir}/*/index.ipynb")
        sys.exit(1)

    results = []
    skipped = []

    for nb_path in notebooks:
        result = process_notebook(nb_path, dry_run=args.dry_run)
        if result is None:
            skipped.append(nb_path.parent.name)
        else:
            results.append(result)

    # Print summary
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}PEP 723 Injection Summary")
    print("=" * 60)
    print(f"Notebooks processed: {len(results)}")
    print(f"Notebooks skipped:   {len(skipped)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    print(f"\n{'Slug':<30} {'Deps':>4}  {'Pyodide':>7}  Dependencies")
    print("-" * 80)
    for r in results:
        pyodide = "yes" if r["pyodide_compatible"] else "NO"
        deps_str = ", ".join(r["dependencies"])
        print(f"{r['slug']:<30} {len(r['dependencies']):>4}  {pyodide:>7}  {deps_str}")

    # Summary of unique dependencies
    all_deps = set()
    for r in results:
        all_deps.update(r["dependencies"])
    print(f"\nUnique PyPI packages ({len(all_deps)}): {', '.join(sorted(all_deps))}")

    incompatible = [r["slug"] for r in results if not r["pyodide_compatible"]]
    if incompatible:
        print(f"\nPyodide-incompatible ({len(incompatible)}): {', '.join(incompatible)}")


if __name__ == "__main__":
    main()
