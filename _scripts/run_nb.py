#!/usr/bin/env python3
"""Quick-test a notebook's code cells sequentially, report first failure.

Usage:
    python _scripts/run_nb.py posts/probability/index.ipynb

This does NOT capture outputs — it just checks if the code runs without errors.
For full execution with output capture, use execute_notebook.py instead.

Skips: PEP 723 metadata cells, magics (%), shell commands (!), Quarto directives (#|).
"""
import json, sys
import matplotlib
matplotlib.use('Agg')

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'index.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

ns = {}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # Skip PEP 723 metadata cells
    if '# ///' in src:
        continue
    # Remove magics and shell commands
    lines = []
    for line in src.split('\n'):
        if line.startswith('%') or line.startswith('!'):
            continue
        if line.startswith('#|'):
            continue
        lines.append(line)
    code = '\n'.join(lines)
    if not code.strip():
        continue
    try:
        exec(compile(code, f'<cell-{i}>', 'exec'), ns)
    except Exception as e:
        print(f'CELL {i} FAILED: {type(e).__name__}: {e}')
        print(f'CODE:\n{code[:500]}')
        sys.exit(1)

print('ALL CELLS PASSED')
