# Skill: Execute a Notebook (Refresh Outputs)

Execute a notebook in-place to refresh all cell outputs (plots, text, tables, etc.) so they reflect the current code. This is essential before deploying since Quarto renders the stored outputs.

## Invocation

```
/execute-notebook <post_path>
```

- `post_path`: path to the notebook (e.g. `posts/probability/index.ipynb`)

## When to Use

- After importing a notebook from the AM207 wiki (outputs may be stale or from Python 2 era)
- After fixing code in a notebook (API migrations, bug fixes, modernization)
- After porting a pymc3 notebook to modern pymc (see `/port-pymc3`)
- Before deploying — ensures rendered site shows current outputs
- When cell outputs are missing or corrupted

## Procedure

### 1. Execute the Notebook

```bash
uv run _scripts/execute_notebook.py <post_path>
```

This script:
1. Reads the notebook's PEP 723 dependencies from its metadata cell
2. Detects any missing packages in the current environment
3. Re-launches itself via `uv run --with <deps>` to bootstrap the right environment
4. Uses `nbclient` to execute every cell and capture outputs back into the `.ipynb`
5. Writes the notebook in-place with fresh outputs

### 2. Options

```bash
# Increase timeout for slow notebooks (pymc sampling, neural network training):
uv run _scripts/execute_notebook.py --timeout 1200 <post_path>

# Continue execution even if a cell errors (captures error output):
uv run _scripts/execute_notebook.py --allow-errors <post_path>
```

### 3. Verify Outputs

After execution, spot-check that outputs were captured:

```bash
python3 -c "
import json
nb = json.load(open('<post_path>'))
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and c.get('outputs')]
print(f'{len(code_cells)} code cells with outputs')
for i, c in code_cells[:5]:
    outs = c['outputs']
    types = [o['output_type'] for o in outs]
    has_img = any('image/png' in str(o.get('data', {})) for o in outs)
    print(f'  Cell {i}: {types} image={has_img}')
"
```

### 4. Quick-Test Alternative

For a fast pass/fail check without output capture (useful for debugging):

```bash
uv run --with <deps> python _scripts/run_nb.py <post_path>
```

This runs cells via `exec()` and reports the first failure, but doesn't update the notebook file.

### 5. Batch Execution (All Notebooks)

```bash
for nb in posts/*/index.ipynb; do
    echo "=== $nb ==="
    uv run _scripts/execute_notebook.py --timeout 1200 "$nb" || echo "FAILED: $nb"
done
```

## Known Slow Notebooks

These take >60s due to MCMC sampling or neural network training:
- `switchpoint` (~90s) — PyMC MCMC sampling
- `utilityorrisk` (~70s) — PyMC MCMC sampling
- `mlp_classification` (~90s) — PyTorch training
- `nnreg` (~80s) — PyTorch training
- `samplingclt` (~230s) — heavy Monte Carlo simulation
- `gibbsconj` (~50s) — Gibbs sampling
- `tetchygibbs` (~40s) — Gibbs sampling

Use `--timeout 1200` for these.

## Notes

- The script is self-bootstrapping via PEP 723 — no pre-built environment needed
- It reads each notebook's own PEP 723 deps to build the right environment
- Do NOT list `arviz` explicitly in PEP 723 deps alongside `pymc` — arviz 1.0 breaks pymc; let pymc pull in the compatible version
- After execution, the notebook's kernel metadata may change (e.g. to `python3`). This is fine.
- Quarto renders stored cell outputs, so notebooks MUST be executed before deploy for fresh results
