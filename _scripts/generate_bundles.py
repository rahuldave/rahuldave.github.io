#!/usr/bin/env python3
"""Generate downloadable zip bundles for notebook posts.

Each notebook post gets a <slug>.zip containing:
  - index.ipynb (source notebook with PEP 723 deps)
  - assets/     (images, CSVs in assets/)
  - data/       (data files if present)

Also generates _site/bundles.json manifest.

Usage:
    python3 _scripts/generate_bundles.py [--site-dir _site] [--posts-dir posts]
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


# Extensions to include from assets/ and data/ directories
BUNDLE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",  # images
    ".csv", ".tsv", ".dat", ".txt", ".json", ".geojson",  # data
    ".npy", ".npz", ".pickle", ".pkl",  # binary data
}


def has_python_code(nb_path: Path) -> bool:
    """Check if notebook has at least one Python code cell."""
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Skip empty cells and PEP 723 metadata-only cells
            stripped = source.strip()
            if stripped and not (stripped.startswith("# /// script") and stripped.endswith("# ///")):
                return True
    return False


def get_pep723_deps(nb_path: Path) -> list[str]:
    """Extract PEP 723 dependencies from notebook."""
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "# /// script" not in source:
            continue
        deps = []
        for line in source.splitlines():
            line = line.strip()
            if line.startswith('#   "') and line.endswith('",'):
                dep = line[5:-2]  # extract package name
                deps.append(dep)
        return deps
    return []


def get_frontmatter(nb_path: Path) -> dict:
    """Extract frontmatter fields from notebook raw cell as a dict."""
    with open(nb_path) as f:
        nb = json.load(f)
    if not nb.get("cells"):
        return {}
    cell = nb["cells"][0]
    if cell.get("cell_type") != "raw":
        return {}
    source = "".join(cell.get("source", []))
    result = {}
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("title:"):
            result["title"] = stripped[6:].strip().strip('"').strip("'")
        elif stripped.startswith("browser-runnable:"):
            val = stripped[len("browser-runnable:"):].strip()
            result["browser-runnable"] = val.lower() == "true"
    return result


def get_notebook_title(nb_path: Path) -> str:
    """Extract title from notebook frontmatter raw cell."""
    fm = get_frontmatter(nb_path)
    return fm.get("title", nb_path.parent.name)


SITE_URL = "https://rahuldave.github.io"


def build_readme(slug: str, title: str, deps: list[str]) -> str:
    """Build a README.md for the bundle."""
    lines = [
        f"# {title}\n",
        "\n",
        f"From <{SITE_URL}/posts/{slug}/>\n",
        "\n",
        "## Quick start\n",
        "\n",
        "```bash\n",
        "# Install uv (if you don't have it)\n",
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n",
        "\n",
        "# Run the notebook (installs dependencies automatically)\n",
        "uvx juv run index.ipynb\n",
        "```\n",
        "\n",
        "`juv` reads the PEP 723 metadata inside the notebook and installs\n",
        "the required packages into an isolated environment.\n",
        "\n",
        "## Dependencies\n",
        "\n",
    ]
    for dep in deps:
        lines.append(f"- {dep}\n")
    lines.append("\n")
    lines.append("## License\n")
    lines.append("\n")
    lines.append("Educational content from Harvard AM207.\n")
    return "".join(lines)


def collect_bundle_files(post_dir: Path) -> list[tuple[Path, str]]:
    """Collect files to include in the zip bundle.

    Returns list of (absolute_path, archive_name) tuples.
    """
    files = []

    # The notebook itself
    nb_path = post_dir / "index.ipynb"
    if nb_path.exists():
        files.append((nb_path, "index.ipynb"))

    # assets/ directory
    assets_dir = post_dir / "assets"
    if assets_dir.is_dir():
        for f in sorted(assets_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in BUNDLE_EXTENSIONS:
                files.append((f, f"assets/{f.name}"))

    # data/ directory
    data_dir = post_dir / "data"
    if data_dir.is_dir():
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in BUNDLE_EXTENSIONS:
                files.append((f, f"data/{f.name}"))

    return files


def generate_bundle(post_dir: Path, site_dir: Path) -> dict | None:
    """Generate zip bundle for a single notebook post.

    Returns metadata dict or None if skipped.
    """
    nb_path = post_dir / "index.ipynb"
    if not nb_path.exists():
        return None

    if not has_python_code(nb_path):
        return None

    slug = post_dir.name
    files = collect_bundle_files(post_dir)

    if not files:
        return None

    # Ensure output directory exists
    out_dir = site_dir / "posts" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / f"{slug}.zip"

    deps = get_pep723_deps(nb_path)
    title = get_notebook_title(nb_path)
    readme = build_readme(slug, title, deps)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arc_name in files:
            zf.write(abs_path, arc_name)
        zf.writestr("README.md", readme)
    # Determine Pyodide compatibility:
    # 1. Explicit frontmatter field takes priority
    # 2. Fall back to auto-detection from dependencies
    fm = get_frontmatter(nb_path)
    if "browser-runnable" in fm:
        pyodide_ok = fm["browser-runnable"]
    else:
        pyodide_incompatible = {"torch", "pymc3", "theano-pymc", "theano", "pymc", "lxml"}
        pyodide_ok = not any(d in pyodide_incompatible for d in deps)

    return {
        "slug": slug,
        "zip": f"/posts/{slug}/{slug}.zip",
        "size_bytes": zip_path.stat().st_size,
        "dependencies": deps,
        "pyodide_compatible": pyodide_ok,
        "files": [arc_name for _, arc_name in files],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate notebook zip bundles")
    parser.add_argument("--site-dir", default="_site", help="Site output directory")
    parser.add_argument("--posts-dir", default="posts", help="Posts source directory")
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    posts_dir = Path(args.posts_dir)

    if not site_dir.exists():
        print(f"Error: site directory '{site_dir}' does not exist. Run `quarto render` first.")
        sys.exit(1)

    notebooks = sorted(posts_dir.glob("*/index.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {posts_dir}/*/index.ipynb")
        sys.exit(1)

    manifest = {}
    skipped = []

    for nb_path in notebooks:
        post_dir = nb_path.parent
        result = generate_bundle(post_dir, site_dir)
        if result is None:
            skipped.append(post_dir.name)
        else:
            slug = result.pop("slug")
            manifest[slug] = result

    # Write manifest
    manifest_path = site_dir / "bundles.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Print summary
    print(f"\nBundle Generation Summary")
    print("=" * 60)
    print(f"Bundles created: {len(manifest)}")
    print(f"Skipped:         {len(skipped)}")

    total_size = sum(v["size_bytes"] for v in manifest.values())
    print(f"Total size:      {total_size / 1024:.0f} KB ({total_size / 1024 / 1024:.1f} MB)")

    print(f"\n{'Slug':<30} {'Size':>8}  {'Files':>5}  {'Deps':>4}  {'Pyodide':>7}")
    print("-" * 70)
    for slug, meta in sorted(manifest.items()):
        size = f"{meta['size_bytes'] / 1024:.0f}K"
        pyodide = "yes" if meta["pyodide_compatible"] else "NO"
        print(f"{slug:<30} {size:>8}  {len(meta['files']):>5}  {len(meta['dependencies']):>4}  {pyodide:>7}")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
