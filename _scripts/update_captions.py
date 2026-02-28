#!/usr/bin/env python3
"""Update image captions in notebook markdown cells.

Takes a notebook path and one or more old->new replacement pairs.
Only modifies markdown cells, leaves code cells untouched.

Usage:
    python3 _scripts/update_captions.py <notebook_path> \
        --replace '![](assets/foo.png)' '![My caption](assets/foo.png)' \
        --replace '![old](assets/bar.png)' '![New caption](assets/bar.png)'

Or pass replacements via JSON file:
    python3 _scripts/update_captions.py <notebook_path> \
        --json replacements.json

JSON format: [["old_string", "new_string"], ...]
"""

import argparse
import json
import sys


def update_captions(notebook_path, replacements):
    """Update image captions in a notebook's markdown cells.

    Args:
        notebook_path: Path to the .ipynb file.
        replacements: List of (old_string, new_string) tuples.

    Returns:
        Number of replacements made.
    """
    with open(notebook_path) as f:
        nb = json.load(f)

    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        for old, new in replacements:
            if old in src:
                src = src.replace(old, new)
                changed += 1
        # Re-split into lines preserving newlines
        lines = src.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)

    return changed


def main():
    parser = argparse.ArgumentParser(description="Update image captions in notebook markdown cells")
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument(
        "--replace",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Replace OLD with NEW in markdown cells (repeatable)",
    )
    parser.add_argument(
        "--json",
        dest="json_file",
        help="JSON file with [[old, new], ...] replacements",
    )
    args = parser.parse_args()

    replacements = []
    if args.replace:
        replacements.extend(args.replace)
    if args.json_file:
        with open(args.json_file) as f:
            replacements.extend(json.load(f))

    if not replacements:
        print("No replacements specified. Use --replace or --json.", file=sys.stderr)
        sys.exit(1)

    changed = update_captions(args.notebook, replacements)
    print(f"{args.notebook}: {changed} caption(s) updated")


if __name__ == "__main__":
    main()
