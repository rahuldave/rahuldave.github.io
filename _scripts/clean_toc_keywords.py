#!/usr/bin/env python3
"""Remove leftover TOC cells and Keywords lines from imported notebooks."""

import json
import re
import sys
from pathlib import Path

POSTS_DIR = Path(__file__).parent.parent / "posts"

# Pattern for the Jekyll TOC block
TOC_PATTERN = re.compile(
    r'^\s*##\s+Contents\s*\n\s*\{:\s*\.no_toc\s*\}\s*\n\s*\*\s*\n\s*\{:\s*toc\s*\}\s*$',
    re.MULTILINE
)

# Pattern for the Keywords line
KEYWORDS_PATTERN = re.compile(r'^#{1,6}\s+Keywords:.*$', re.MULTILINE)


def clean_notebook(nb_path: Path) -> bool:
    """Clean TOC and Keywords from a notebook. Returns True if modified."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    modified = False
    cells_to_remove = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'markdown':
            continue

        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Check for TOC cell - remove entire cell if it's just the TOC block
        stripped = source.strip()
        if TOC_PATTERN.match(stripped):
            cells_to_remove.append(i)
            modified = True
            print(f"  Cell {i}: removing TOC cell")
            continue

        # Check for Keywords line - remove the line from the cell
        if KEYWORDS_PATTERN.search(source):
            new_source = KEYWORDS_PATTERN.sub('', source)
            # Clean up: remove resulting blank lines at start/end
            lines = new_source.split('\n')
            # Remove leading blank lines
            while lines and lines[0].strip() == '':
                lines.pop(0)
            # Remove trailing blank lines (keep at most one)
            while len(lines) > 1 and lines[-1].strip() == '' and lines[-2].strip() == '':
                lines.pop()
            new_source = '\n'.join(lines)

            if new_source.strip() == '':
                # Cell is now empty, remove it
                cells_to_remove.append(i)
                print(f"  Cell {i}: removing empty cell (was Keywords only)")
            else:
                # Update cell source
                cell['source'] = new_source.split('\n')
                # Re-add newlines between lines (json notebook format)
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print(f"  Cell {i}: removed Keywords line")
            modified = True

    # Remove cells in reverse order to preserve indices
    for i in sorted(cells_to_remove, reverse=True):
        nb['cells'].pop(i)

    if modified:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')

    return modified


def main():
    notebooks = sorted(POSTS_DIR.glob('*/index.ipynb'))
    changed = 0
    for nb_path in notebooks:
        print(f"Checking {nb_path.parent.name}/index.ipynb...")
        if clean_notebook(nb_path):
            changed += 1
            print(f"  -> CLEANED")
        else:
            print(f"  -> ok")

    print(f"\nDone. {changed} notebooks cleaned.")


if __name__ == '__main__':
    main()
