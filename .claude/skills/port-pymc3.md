# Skill: Port PyMC3 Notebook to Modern PyMC

## When to Use
When a notebook imports `pymc3` or `theano` and needs to be updated to modern `pymc` (v5+) which uses `pytensor` as its backend.

## Porting Reference
Full reference at `_scripts/pymc3-to-pymc-porting.md`. Key changes below.

## Step-by-Step Process

### 1. Check if pymc3 is actually used
Some notebooks import pymc3 but never use it. Check:
```
grep -c "pm\." posts/SLUG/index.ipynb
```
If only `import pymc3 as pm` with no `pm.` calls, just remove the import and pymc3 from PEP 723 deps.

### 2. Update imports
```python
# OLD                                    # NEW
import pymc3 as pm                       import pymc as pm
                                         import arviz as az
from pymc3.math import switch            from pymc.math import switch
import theano.tensor as tt               import pytensor.tensor as pt
```

### 3. Fix distribution parameters
- `sd=` → `sigma=` (Normal, HalfNormal, Lognormal, StudentT, etc.)
- `testval=` → `initval=`

### 4. Fix sampling calls
```python
# OLD
trace = pm.sample(40000, njobs=4)
burned = trace[5000:]
burned["varname"]  # numpy array

# NEW
idata = pm.sample(40000, cores=4)
burned = idata.sel(draw=slice(5000, None))
burned.posterior["varname"].values.flatten()  # numpy array
```

Do NOT use `return_inferencedata=False` — use InferenceData throughout.

### 5. Fix trace slicing patterns
| pymc3 pattern | pymc5 pattern |
|---|---|
| `trace[N:]` | `idata.sel(draw=slice(N, None))` |
| `trace[N::step]` | `idata.sel(draw=slice(N, None, step))` |
| `trace["var"]` | `idata.posterior["var"].values.flatten()` |

### 6. Move plotting/diagnostics to arviz
```python
# OLD                              # NEW
pm.traceplot(trace)                az.plot_trace(idata)
pm.forestplot(trace)               az.plot_forest(idata)
pm.autocorrplot(trace)             az.plot_autocorr(idata)
pm.summary(trace)                  az.summary(idata)
pm.gelman_rubin(trace)             az.rhat(idata)
pm.effective_n(trace)              az.ess(idata)
pm.geweke(trace)                   # REMOVED — use az.rhat(), az.ess()
```

### 7. Fix other API changes
```python
# OLD                              # NEW
model.vars                         model.free_RVs
rv.random(size=N)                  pm.draw(rv, draws=N)
pm.sample_ppc(trace, samples=N)    pm.sample_posterior_predictive(idata)
pm.trace_to_dataframe(trace)       idata.posterior.to_dataframe()
```

### 8. Wrap graphviz calls
```python
try:
    pm.model_to_graphviz(model)
except ImportError:
    print("graphviz not available, skipping model visualization")
```

### 9. Wrap introspective cells in try/except
Cells that explore pymc internals (`.distribution`, `.transformed`, `.logp()`, `model['var_log__']`) have changed significantly. Wrap in try/except:
```python
try:
    model['early_mean_log__']
except (KeyError, TypeError):
    print('Transformed variable access has changed in modern pymc')
```

### 10. Update PEP 723 dependencies
```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pymc",
#   "scipy",
#   "seaborn",
# ]
# ///
```

**CRITICAL**: Do NOT list `arviz` explicitly! arviz 1.0.0 breaks `from arviz import InferenceData`. pymc constrains arviz to <1.0 and pulls in the correct version.

Remove any `requires-python` upper bounds (no `<3.13` etc.).

### 11. Fix scipy.optimize.fmin return
Modern numpy is stricter about assigning arrays to scalar positions:
```python
# OLD
opt_predictions[i] = fmin(tomin, 0, disp=False)

# NEW
opt_predictions[i] = fmin(tomin, 0, disp=False).item()
```

### 12. Test
```bash
# Quick test (no output capture):
uv run --with pymc --with matplotlib --with numpy --with scipy --with seaborn --with pandas \
    python _scripts/run_nb.py posts/SLUG/index.ipynb

# Full test via juv bundle:
python _scripts/generate_bundles.py
python _scripts/test_bundles.py --slug SLUG --timeout 600

# Execute with output capture:
uv run _scripts/execute_notebook.py posts/SLUG/index.ipynb
```

## Notebooks Already Ported
- `em` — removed unused pymc3 import (no actual pymc usage)
- `hmcexplore` — removed unused pymc3 import (no actual pymc usage)
- `utilityorrisk` — full port to InferenceData API
- `switchpoint` — full port to InferenceData API, arviz diagnostics

## Notebooks Coming (Lectures 18–26)
Many more pymc3 notebooks are expected from the AM207 wiki import. Follow this same process.
