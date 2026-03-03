# PyMC3 → PyMC (v5+) Porting Reference

## Imports
```python
# OLD (pymc3)
import pymc3 as pm
from pymc3.math import switch, exp, log
import theano.tensor as tt

# NEW (pymc v5)
import pymc as pm
import arviz as az
from pymc.math import switch, exp, log
import pytensor.tensor as pt
```

## Distribution Parameters
- `sd=` → `sigma=` (Normal, HalfNormal, Lognormal, etc.)
  ```python
  # OLD: pm.Normal("x", mu=0, sd=1)
  # NEW: pm.Normal("x", mu=0, sigma=1)
  ```
- `testval=` → `initval=`
- `shape=` → `shape=` or use `dims=` with coords (preferred)
- `lam=` still works for Exponential, Poisson

## Sampling
```python
# OLD
trace = pm.sample(40000, njobs=4)
burned = trace[5000:]          # slice for burn-in
vals = burned["varname"]       # numpy array

# NEW (returns InferenceData by default)
idata = pm.sample(40000, cores=4)
burned = idata.sel(draw=slice(5000, None))   # xarray-style
vals = burned.posterior["varname"].values.flatten()  # numpy array
```

### Key pm.sample() changes
| pymc3 | pymc5 | Notes |
|-------|-------|-------|
| `njobs=` | `cores=` | |
| returns MultiTrace | returns InferenceData | Use `return_inferencedata=False` for old behavior |
| `trace[N:]` | `idata.sel(draw=slice(N, None))` | Label-based xarray slicing |
| `trace["var"]` | `idata.posterior["var"].values.flatten()` | Returns xarray DataArray → .values for numpy |
| `trace[N::step]` | `idata.sel(draw=slice(N, None, step))` | Thinning |

### Extracting numpy arrays from InferenceData
```python
# Single chain, all draws (common pattern):
samples = idata.posterior["varname"].values.flatten()

# Or use arviz extract (stacks chains automatically):
samples = az.extract(idata, var_names=["varname"])["varname"].values
```

## Posterior Predictive
```python
# OLD
pm.sample_ppc(trace, samples=200)  # returns dict

# NEW
pm.sample_posterior_predictive(idata)  # returns InferenceData

# Access values:
ppc = pm.sample_posterior_predictive(idata)
ppc.posterior_predictive["obs"].values  # shape: (chains, draws, obs_shape)
```
- `pm.sample_ppc()` → `pm.sample_posterior_predictive()`
- No `samples=` / `draws=` parameter; it samples from all posterior draws
- Returns InferenceData by default; use `return_inferencedata=False` for dict

## Plotting (all moved to arviz)
```python
# OLD                              # NEW
pm.traceplot(trace)                az.plot_trace(idata)
pm.forestplot(trace)               az.plot_forest(idata)
pm.autocorrplot(trace)             az.plot_autocorr(idata)
pm.plot_posterior(trace)           az.plot_posterior(idata)
pm.summary(trace)                  az.summary(idata)
pm.energyplot(trace)               az.plot_energy(idata)
```
All arviz plot functions accept InferenceData natively.

## Diagnostics
```python
# OLD                                      # NEW
pm.gelman_rubin(trace)                     az.rhat(idata)
pm.effective_n(trace)                      az.ess(idata)
pm.geweke(trace)                           # REMOVED (use az.rhat, az.ess, az.mcse)
pm.summary(trace)                          az.summary(idata)
```

## Model Attributes
```python
# OLD                              # NEW
model.vars                         model.free_RVs
model.named_vars                   # removed; use model.free_RVs + model.observed_RVs + model.deterministics
model.deterministics               model.deterministics  # still works
model.observed_RVs                 model.observed_RVs    # still works
model.basic_RVs                    model.basic_RVs       # still works
model.unobserved_RVs               model.unobserved_RVs  # still works
model['varname_log__']             # transformed variable access changed; use model.rvs_to_transforms
```

## Variable Methods
```python
# OLD                              # NEW
rv.random(size=1000)               pm.draw(rv, draws=1000)
rv.logp(point_dict)                model.point_logps()  # or model.compile_logp()
rv.distribution.defaults           # removed
rv.transformed                     # use model.rvs_to_transforms
```

## Trace Utilities
```python
# OLD                              # NEW
pm.trace_to_dataframe(trace)       idata.posterior.to_dataframe()
az.from_pymc3(trace)               # deprecated; use InferenceData directly
```

## PEP 723 Dependencies (for juv bundles)
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
**IMPORTANT**: Do NOT list `arviz` explicitly. `pymc` depends on `arviz<1.0` (0.23.x).
If `arviz` is listed separately, uv may resolve it to v1.0.0 (major refactor that breaks
`from arviz import InferenceData`). Let pymc pull in the compatible version.

No python upper bound needed. No numpy pin needed. `pymc` pulls in `pytensor` automatically.

## Common Patterns

### Burn-in + plotting
```python
# OLD
trace = pm.sample(40000)
burned = trace[5000::5]
pm.traceplot(burned)

# NEW
idata = pm.sample(40000)
burned = idata.sel(draw=slice(5000, None, 5))
az.plot_trace(burned)
```

### Extract samples for computation
```python
# OLD
alpha = trace[5000:]["alpha"]  # numpy array

# NEW
alpha = idata.posterior["alpha"].sel(draw=slice(5000, None)).values.flatten()
# or: alpha = az.extract(idata.sel(draw=slice(5000, None)))["alpha"].values
```

### Posterior predictive with dict access
```python
# OLD
ppc = pm.sample_ppc(trace, samples=200)
ppc["obs"].shape  # (200, n_obs)

# NEW
ppc = pm.sample_posterior_predictive(idata)
# shape is (chains, draws, n_obs) in InferenceData:
ppc_vals = ppc.posterior_predictive["obs"].values
# flatten chains: ppc_vals.reshape(-1, ppc_vals.shape[-1])
```

### Model visualization (graphviz)
```python
# graphviz requires system binary; wrap for portability:
try:
    pm.model_to_graphviz(model)
except ImportError:
    print("graphviz not available")
```

## Notebooks Ported So Far
- `em` — removed unused pymc3 import (no actual pymc usage)
- `hmcexplore` — removed unused pymc3 import (no actual pymc usage)
- `utilityorrisk` — full port: pymc3→pymc, sd→sigma, InferenceData patterns
- `switchpoint` — full port: pymc3→pymc, InferenceData, arviz diagnostics

## Known Issues
- `az.from_pymc3()` is deprecated and may not work with modern pymc's MultiTrace
- `graphviz` Python package alone isn't enough; needs `dot` binary (not pip-installable)
- HMC notebooks with large N (>10000) may timeout in 300s bundle tests
