#!/usr/bin/env python3
"""Add format: {html: default, ipynb: default} to notebook frontmatter.

Finds all posts/*/index.ipynb files, checks the first raw cell for
Quarto YAML frontmatter, and inserts the format block if missing.

Usage:
    python3 _scripts/add_notebook_format.py          # dry run
    python3 _scripts/add_notebook_format.py --write   # actually modify files
"""

import argparse
import glob
import json
import sys

FORMAT_LINES = [
    "format:\n",
    "    html: default\n",
    "    ipynb: default\n",
]


def add_format_to_frontmatter(source_lines):
    """Insert format block into frontmatter source lines.

    Handles both formats: source as a list of individual lines,
    or source as a single string with embedded newlines.

    Returns (new_lines, changed) where changed is True if modification was made.
    """
    text = "".join(source_lines)
    if "\nformat:" in text or text.startswith("format:"):
        return source_lines, False

    # Split into actual lines, preserving newlines
    lines = text.splitlines(keepends=True)
    # Last line may lack trailing newline (e.g. "---")
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    # Find closing --- and insert format block before it
    new_lines = []
    inserted = False
    for line in lines:
        if line.rstrip() == "---" and not inserted and new_lines:
            new_lines.extend(FORMAT_LINES)
            inserted = True
        new_lines.append(line)

    if not inserted:
        return source_lines, False

    return new_lines, True


def main():
    parser = argparse.ArgumentParser(description="Add format frontmatter to notebook posts")
    parser.add_argument("--write", action="store_true", help="Actually modify files (default: dry run)")
    parser.add_argument("--posts-dir", default="posts", help="Posts directory (default: posts)")
    args = parser.parse_args()

    notebooks = sorted(glob.glob(f"{args.posts_dir}/*/index.ipynb"))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    modified = 0
    skipped = 0
    errors = 0

    for nb_path in notebooks:
        with open(nb_path) as f:
            nb = json.load(f)

        if not nb.get("cells"):
            print(f"  SKIP (no cells): {nb_path}")
            skipped += 1
            continue

        cell = nb["cells"][0]
        if cell.get("cell_type") != "raw":
            print(f"  SKIP (first cell not raw): {nb_path}")
            skipped += 1
            continue

        source = cell["source"]
        text = "".join(source)
        if not text.startswith("---"):
            print(f"  SKIP (no YAML frontmatter): {nb_path}")
            skipped += 1
            continue

        new_source, changed = add_format_to_frontmatter(source)
        if not changed:
            print(f"  SKIP (already has format): {nb_path}")
            skipped += 1
            continue

        if args.write:
            nb["cells"][0]["source"] = new_source
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1)
            print(f"  UPDATED: {nb_path}")
        else:
            print(f"  WOULD UPDATE: {nb_path}")
        modified += 1

    print(f"\n{'Modified' if args.write else 'Would modify'}: {modified}, Skipped: {skipped}, Errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
