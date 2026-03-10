---
name: edit-notebook
description: >
  Atomic tools for reading and editing Jupyter notebook cells.
  Use these instead of writing inline Python scripts to manipulate notebook JSON.
  The underlying library (_scripts/notebook_tools.py) is also importable by other scripts.
user-invocable: false
allowed-tools: Bash(python3 _scripts/notebook_tools.py *)
---

# Notebook Cell Editing Tools

**ALWAYS use `_scripts/notebook_tools.py` instead of inline `python3 << 'PYEOF'` scripts when editing notebooks.**

All commands:
```bash
python3 _scripts/notebook_tools.py <command> <notebook_path> [args...]
```

## Commands

### list — Show cell overview

```bash
python3 _scripts/notebook_tools.py list <notebook> [--type code|markdown|raw] [--lines N]
```

Shows cell index, type, line count, and preview for each cell. Use `--type` to filter and `--lines` to show more preview lines.

### read — Read a single cell

```bash
python3 _scripts/notebook_tools.py read <notebook> <cell_index>
```

Shows full cell source with line numbers. Use this before editing to see exact content.

### edit — String replacement in a cell

```bash
python3 _scripts/notebook_tools.py edit <notebook> <cell_index> --old 'OLD_STRING' --new 'NEW_STRING' [--replace-all]
```

Like the Edit tool but for a notebook cell. Fails if `--old` is not found or is ambiguous (unless `--replace-all`). **Saves the notebook automatically.**

For multi-line strings, use a heredoc:
```bash
python3 _scripts/notebook_tools.py edit <notebook> <cell_index> \
  --old "$(cat <<'EOF'
old text
spanning lines
EOF
)" \
  --new "$(cat <<'EOF'
new text
spanning lines
EOF
)"
```

### replace — Replace entire cell source

```bash
python3 _scripts/notebook_tools.py replace <notebook> <cell_index> --source 'NEW_SOURCE' [--type code|markdown|raw]
```

Replaces the full source of a cell. Optionally changes cell type. **Saves automatically.**

For multi-line source, use a heredoc:
```bash
python3 _scripts/notebook_tools.py replace <notebook> <cell_index> --source "$(cat <<'EOF'
import pymc as pm
import arviz as az
EOF
)"
```

### add — Insert a new cell

```bash
python3 _scripts/notebook_tools.py add <notebook> <cell_index> --position above|below --type code|markdown|raw --source 'SOURCE'
```

Inserts a new cell above or below the given index. **Saves automatically.**

### delete — Remove cells

```bash
python3 _scripts/notebook_tools.py delete <notebook> <cell_index> [<cell_index2> ...]
```

Deletes one or more cells. Indices are processed highest-first to preserve correctness. **Saves automatically.**

### find — Search cells with regex

```bash
python3 _scripts/notebook_tools.py find <notebook> <pattern> [--type code|markdown|raw] [-C N] [-i]
```

Regex search across cells. Shows matching cell indices and lines. Use `-C` for context lines, `-i` for case-insensitive, `--type` to filter.

## When to Use

**ALWAYS use these tools instead of inline Python scripts when:**
- Inspecting notebook structure (`list`, `read`)
- Making targeted edits to cell source (`edit`)
- Replacing whole cells during porting (`replace`)
- Adding or removing cells (`add`, `delete`)
- Searching for patterns across cells (`find`)

**Common workflows:**

1. **Porting pymc3 → pymc**: `find` for patterns, then `edit` or `replace` each cell
2. **Removing wiki artifacts**: `read` cell 1, then `delete` if it's H1/keywords
3. **Adding captions**: `find` for `![](` patterns, then `edit` to add alt text
4. **Injecting cells**: `add` for new cells (e.g., imports, PEP 723 metadata)

## Key Behaviors

- `edit`, `replace`, `add`, `delete` all **save the notebook automatically** after the operation
- `list`, `read`, `find` are **read-only** and never modify the file
- Cell indices start at 0
- After `add` or `delete`, indices shift — re-run `list` if you need to make further changes
- The script handles source as line-lists internally (proper `.ipynb` format)
- New cells get a random `id` field (satisfies nbformat requirements)

## Practical Notes

- **Prefer `notebook_tools.py` over the `NotebookEdit` tool** for wiki-imported notebooks — they lack cell IDs, which `NotebookEdit` requires
- **Index shift after PEP 723 injection**: `inject_juv_metadata.py` inserts a cell at index 1, so all subsequent cell indices shift by +1. Account for this when editing cells after injection.
- **Subagents cannot use these tools** due to permission restrictions — do notebook edits in the main conversation
- **For multi-line `replace` with quotes**: use shell quoting `'"'"'` for embedded single quotes, or heredoc syntax

## Library Usage (for other scripts)

```python
from notebook_tools import load_notebook, save_notebook, get_source, set_source, make_cell
nb = load_notebook("posts/foo/index.ipynb")
set_source(nb["cells"][5], "import pymc as pm")
save_notebook("posts/foo/index.ipynb", nb)
```
