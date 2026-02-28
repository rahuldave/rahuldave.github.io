#!/usr/bin/env python3
"""Import a wiki notebook (.ipynb) as a Quarto blog post.

Creates posts/<name>/index.ipynb with a raw frontmatter cell prepended,
fixes image paths (images/ -> assets/), and copies referenced images.

Usage:
    python3 _scripts/import_notebook.py \
        --source ~/Attic/Projects/AM207/2018fall_wiki/wiki/MLE.ipynb \
        --name MLE \
        --title "Maximum Likelihood Estimation" \
        --subtitle "Find the parameters that make your data most probable." \
        --description "Two-sentence description here." \
        --categories statistics models \
        --date 2025-02-26 \
        --images gaussmle.png linregmle.png

The --images flag copies files from the same directory as --source's
sibling images/ folder into posts/<name>/assets/.
"""

import argparse
import json
import shutil
from pathlib import Path


def build_frontmatter_cell(title, subtitle, description, categories, date):
    """Build a Quarto raw YAML frontmatter cell."""
    lines = [
        "---\n",
        f'title: "{title}"\n',
        f'subtitle: "{subtitle}"\n',
        f'description: "{description}"\n',
        "categories:\n",
    ]
    for cat in categories:
        lines.append(f"    - {cat}\n")
    lines.append(f"date: {date}\n")
    lines.append("---")
    return {
        "cell_type": "raw",
        "metadata": {},
        "source": lines,
    }


def fix_image_paths(cell):
    """Replace images/ with assets/ in markdown cell source lines."""
    if cell["cell_type"] != "markdown":
        return cell
    new_source = [
        line.replace("](./images/", "](assets/").replace("](images/", "](assets/")
        for line in cell["source"]
    ]
    if new_source != cell["source"]:
        cell = dict(cell)
        cell["source"] = new_source
    return cell


def main():
    parser = argparse.ArgumentParser(description="Import a wiki notebook as a Quarto blog post")
    parser.add_argument("--source", required=True, help="Path to source .ipynb")
    parser.add_argument("--name", required=True, help="Post directory name under posts/")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--images", nargs="*", default=[], help="Image filenames to copy from source images/ dir")
    parser.add_argument("--posts-dir", default="posts", help="Posts directory (default: posts)")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source notebook not found: {source}")

    # Paths
    post_dir = Path(args.posts_dir) / args.name
    assets_dir = post_dir / "assets"
    dest = post_dir / "index.ipynb"

    # Create directories
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Read source notebook
    with open(source) as f:
        nb = json.load(f)

    # Build new cell list: frontmatter + original cells with fixed paths
    frontmatter = build_frontmatter_cell(
        args.title, args.subtitle, args.description, args.categories, args.date
    )
    new_cells = [frontmatter] + [fix_image_paths(c) for c in nb["cells"]]
    nb["cells"] = new_cells

    # Normalize notebook metadata
    nb.setdefault("metadata", {})
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11.0"}
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    # Write notebook
    with open(dest, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {dest}")

    # Copy images
    images_src_dir = source.parent / "images"
    for img_name in args.images:
        img_src = images_src_dir / img_name
        if not img_src.exists():
            print(f"  WARNING: image not found: {img_src}")
            continue
        shutil.copy2(img_src, assets_dir / img_name)
        print(f"  Copied {img_name} -> {assets_dir / img_name}")

    print(f"Done. {len(new_cells)} cells, {len(args.images)} images.")


if __name__ == "__main__":
    main()
