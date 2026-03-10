#!/usr/bin/env python3
"""Notebook cell editing tools for Claude Code.

Provides atomic operations on Jupyter notebook cells:
list, read, edit, replace, add, delete, find.

Usage:
    python _scripts/notebook_tools.py <command> <notebook> [args...]

Commands:
    list    <notebook> [--type code|markdown|raw] [--lines N]
    read    <notebook> <cell_index>
    edit    <notebook> <cell_index> --old OLD --new NEW
    replace <notebook> <cell_index> --source SOURCE [--type code|markdown|raw]
    add     <notebook> <cell_index> --position above|below --type code|markdown|raw --source SOURCE
    delete  <notebook> <cell_index> [<cell_index2> ...]
    find    <notebook> <pattern> [--type code|markdown|raw] [--context N]
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path


def load_notebook(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_notebook(path: str, nb: dict):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def get_source(cell: dict) -> str:
    """Get cell source as a single string."""
    return "".join(cell["source"])


def set_source(cell: dict, text: str):
    """Set cell source from a string, preserving line-list format."""
    if not text:
        cell["source"] = []
        return
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]]
    if lines[-1]:  # non-empty last line
        cell["source"].append(lines[-1])
    elif len(lines) > 1:
        # text ended with \n, last element is empty string after split
        pass  # already handled by the \n on previous lines


def make_cell(cell_type: str, source: str) -> dict:
    """Create a new notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [],
        "id": uuid.uuid4().hex[:8],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    set_source(cell, source)
    return cell


# ─── Commands ───────────────────────────────────────────────────────

def cmd_list(nb: dict, args):
    """List cells with index, type, and first N lines."""
    cells = nb["cells"]
    max_lines = args.lines or 1
    for i, cell in enumerate(cells):
        if args.type and cell["cell_type"] != args.type:
            continue
        src = get_source(cell)
        lines = src.split("\n")
        preview = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += " ..."
        # Truncate long lines
        preview_lines = preview.split("\n")
        preview_lines = [l[:120] + ("..." if len(l) > 120 else "") for l in preview_lines]
        preview = "\n".join(preview_lines)
        line_count = len(lines)
        print(f"[{i}] {cell['cell_type']:8s} ({line_count:3d} lines) | {preview}")


def cmd_read(nb: dict, args):
    """Read full source of a cell, with line numbers."""
    idx = args.cell_index
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell index {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    cell = cells[idx]
    src = get_source(cell)
    print(f"Cell {idx} ({cell['cell_type']}, {len(src.split(chr(10)))} lines):")
    for n, line in enumerate(src.split("\n"), 1):
        print(f"  {n:4d}\t{line}")


def cmd_edit(nb: dict, args):
    """String replacement within a cell (like Edit tool)."""
    idx = args.cell_index
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell index {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    cell = cells[idx]
    src = get_source(cell)
    count = src.count(args.old)
    if count == 0:
        print(f"Error: old string not found in cell {idx}", file=sys.stderr)
        print(f"Cell source:\n{src}", file=sys.stderr)
        sys.exit(1)
    if count > 1 and not args.replace_all:
        print(f"Error: old string found {count} times in cell {idx}. Use --replace-all to replace all.", file=sys.stderr)
        sys.exit(1)
    if args.replace_all:
        new_src = src.replace(args.old, args.new)
    else:
        new_src = src.replace(args.old, args.new, 1)
    set_source(cell, new_src)
    save_notebook(args.notebook, nb)
    print(f"Edited cell {idx}: replaced {count} occurrence(s)")


def cmd_replace(nb: dict, args):
    """Replace entire cell source."""
    idx = args.cell_index
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell index {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    cell = cells[idx]
    if args.type:
        cell["cell_type"] = args.type
        if args.type == "code" and "outputs" not in cell:
            cell["outputs"] = []
            cell["execution_count"] = None
    set_source(cell, args.source)
    save_notebook(args.notebook, nb)
    print(f"Replaced cell {idx} ({cell['cell_type']})")


def cmd_add(nb: dict, args):
    """Insert a new cell above or below a given index."""
    idx = args.cell_index
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell index {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    new_cell = make_cell(args.type, args.source)
    insert_at = idx if args.position == "above" else idx + 1
    cells.insert(insert_at, new_cell)
    save_notebook(args.notebook, nb)
    print(f"Added {args.type} cell at index {insert_at}")


def cmd_delete(nb: dict, args):
    """Delete one or more cells by index."""
    cells = nb["cells"]
    indices = sorted(set(args.cell_indices), reverse=True)  # delete from end to preserve indices
    for idx in indices:
        if idx < 0 or idx >= len(cells):
            print(f"Error: cell index {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
            sys.exit(1)
    for idx in indices:
        cell = cells.pop(idx)
        print(f"Deleted cell {idx} ({cell['cell_type']})")
    save_notebook(args.notebook, nb)


def cmd_find(nb: dict, args):
    """Regex search across cells."""
    cells = nb["cells"]
    pattern = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    context = args.context or 0
    found = False
    for i, cell in enumerate(cells):
        if args.type and cell["cell_type"] != args.type:
            continue
        src = get_source(cell)
        lines = src.split("\n")
        matches = []
        for n, line in enumerate(lines):
            if pattern.search(line):
                matches.append(n)
        if matches:
            found = True
            print(f"[{i}] {cell['cell_type']}:")
            for m in matches:
                start = max(0, m - context)
                end = min(len(lines), m + context + 1)
                for j in range(start, end):
                    marker = ">>>" if j == m else "   "
                    print(f"  {marker} {j+1:4d}\t{lines[j][:120]}")
            print()
    if not found:
        print("No matches found.")


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Notebook cell editing tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List cells")
    p_list.add_argument("notebook")
    p_list.add_argument("--type", choices=["code", "markdown", "raw"])
    p_list.add_argument("--lines", type=int, default=1, help="Lines of preview per cell")

    # read
    p_read = subparsers.add_parser("read", help="Read a cell")
    p_read.add_argument("notebook")
    p_read.add_argument("cell_index", type=int)

    # edit
    p_edit = subparsers.add_parser("edit", help="String replacement in a cell")
    p_edit.add_argument("notebook")
    p_edit.add_argument("cell_index", type=int)
    p_edit.add_argument("--old", required=True, help="String to find")
    p_edit.add_argument("--new", required=True, help="Replacement string")
    p_edit.add_argument("--replace-all", action="store_true", help="Replace all occurrences")

    # replace
    p_replace = subparsers.add_parser("replace", help="Replace entire cell source")
    p_replace.add_argument("notebook")
    p_replace.add_argument("cell_index", type=int)
    p_replace.add_argument("--source", required=True, help="New cell source")
    p_replace.add_argument("--type", choices=["code", "markdown", "raw"], help="Change cell type")

    # add
    p_add = subparsers.add_parser("add", help="Insert a new cell")
    p_add.add_argument("notebook")
    p_add.add_argument("cell_index", type=int, help="Reference cell index")
    p_add.add_argument("--position", choices=["above", "below"], required=True)
    p_add.add_argument("--type", choices=["code", "markdown", "raw"], required=True)
    p_add.add_argument("--source", required=True, help="Cell source content")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete cells")
    p_delete.add_argument("notebook")
    p_delete.add_argument("cell_indices", type=int, nargs="+", help="Cell indices to delete")

    # find
    p_find = subparsers.add_parser("find", help="Search cells with regex")
    p_find.add_argument("notebook")
    p_find.add_argument("pattern", help="Regex pattern")
    p_find.add_argument("--type", choices=["code", "markdown", "raw"])
    p_find.add_argument("--context", "-C", type=int, default=0, help="Context lines")
    p_find.add_argument("--ignore-case", "-i", action="store_true")

    args = parser.parse_args()

    nb = load_notebook(args.notebook)

    commands = {
        "list": cmd_list,
        "read": cmd_read,
        "edit": cmd_edit,
        "replace": cmd_replace,
        "add": cmd_add,
        "delete": cmd_delete,
        "find": cmd_find,
    }

    commands[args.command](nb, args)


if __name__ == "__main__":
    main()
