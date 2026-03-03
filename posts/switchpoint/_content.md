<!-- cell:1 type:code -->
```python
#| include: false

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

<!-- cell:2 type:code -->
```python
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
```

<!-- cell:3 type:markdown -->
## A switchpoint model

This is a model of coal-mine diasaters in England. Somewhere around 1900, regulation was introduced, and in response, miing became safer. But if we were forensically looking at such data, we would be able to detect such change using a switchpoint model. We'd then have to search for the causality.

### Data

<!-- cell:4 type:code -->
```python
disasters_data = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                         2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                         3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

n_years = len(disasters_data)

plt.figure(figsize=(12.5, 3.5))
plt.bar(np.arange(1851, 1962), disasters_data, color="#348ABD")
plt.xlabel("Year")
plt.ylabel("Disasters")
plt.title("UK coal mining disasters, 1851-1962")
plt.xlim(1851, 1962);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-4-output-1.png)

<!-- cell:5 type:markdown -->
One can see the swtich roughly in the picture above.

<!-- cell:6 type:markdown -->
### Model

We'll assume a Poisson model for the mine disasters; appropriate because the counts are low.

$$
y \vert \tau, \lambda_1, \lambda_2 \sim Poisson(r_t)\\
r_t = \lambda_1 \,{\rm if}\, t < \tau \,{\rm else}\, \lambda_2 \,{\rm for}\, t \in [t_l, t_h]\\
\tau \sim DiscreteUniform(t_l, t_h)\\
\lambda_1 \sim Exp(a)\\
\lambda_2 \sim Exp(b)\\
$$

<!-- cell:7 type:markdown -->
The rate parameter varies before and after the switchpoint, which itseld has a discrete-uniform prior on it. Rate parameters get exponential priors.

<!-- cell:8 type:code -->
```python
import pymc as pm
import arviz as az
from pymc.math import switch
with pm.Model() as coaldis1:
    early_mean = pm.Exponential('early_mean', 1)
    late_mean = pm.Exponential('late_mean', 1)
    switchpoint = pm.DiscreteUniform('switchpoint', lower=0, upper=n_years)
    rate = switch(switchpoint >= np.arange(n_years), early_mean, late_mean)
    disasters = pm.Poisson('disasters', mu=rate, observed=disasters_data)

```

<!-- cell:9 type:code -->
```python
try:
    pm.model_to_graphviz(coaldis1)
except ImportError:
    print("graphviz not available, skipping model visualization")
```
Output:
```
graphviz not available, skipping model visualization
```

<!-- cell:10 type:markdown -->
Let us interrogate our model about the various parts of it. Notice that our stochastics are logs of the rate params and the switchpoint, while our deterministics are the rate parameters themselves.

<!-- cell:11 type:code -->
```python
coaldis1.free_RVs #stochastics
```
Output:
```
[early_mean, late_mean, switchpoint]
```

<!-- cell:12 type:code -->
```python
try:
    type(coaldis1['early_mean_log__'])
except (KeyError, TypeError):
    print('Transformed variable access has changed in modern pymc')
```
Output:
```
Transformed variable access has changed in modern pymc
```

<!-- cell:13 type:code -->
```python
coaldis1.deterministics #deterministics
```
Output:
```
[]
```

<!-- cell:14 type:markdown -->
Labelled variables show up in traces, or for predictives. We also list the "likelihood" stochastics.

<!-- cell:15 type:code -->
```python
# In modern pymc, use model.free_RVs, model.observed_RVs, model.deterministics
{rv.name: rv for rv in coaldis1.free_RVs + coaldis1.observed_RVs + coaldis1.deterministics}
```
Output:
```
{'early_mean': early_mean,
 'late_mean': late_mean,
 'switchpoint': switchpoint,
 'disasters': disasters}
```

<!-- cell:16 type:code -->
```python
coaldis1.observed_RVs
```
Output:
```
[disasters]
```

<!-- cell:17 type:markdown -->
The DAG based structure and notation used in pymc3 and similar software makes no distinction between random variables and data. Everything is a node, and some nodes are conditioned upon. This is reminiscent of the likelihood being considered a function of its parameters. But you can consider it as a function of data with fixed parameters and sample from it.

<!-- cell:18 type:markdown -->
You can sample from the distributions in `pymc3`.

<!-- cell:19 type:code -->
```python
with coaldis1:
    plt.hist(pm.draw(coaldis1['switchpoint'], draws=1000));
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-12-output-1.png)

<!-- cell:20 type:code -->
```python
try:
    print(early_mean.transformed, switchpoint.distribution)
except AttributeError:
    print('Internal distribution API has changed in modern pymc')
```
Output:
```
Internal distribution API has changed in modern pymc
```

<!-- cell:21 type:code -->
```python
try:
    switchpoint.distribution.defaults
except AttributeError:
    print('Distribution defaults API has changed in modern pymc')
```
Output:
```
Distribution defaults API has changed in modern pymc
```

<!-- cell:22 type:code -->
```python
ed=pm.Exponential.dist(1)
print(type(ed))
pm.draw(ed, draws=10)
```
Output:
```
<class 'pytensor.tensor.variable.TensorVariable'>
```
Output:
```
array([1.06307892, 4.1061264 , 1.01125516, 0.04178897, 0.03174314,
       1.36572425, 0.77706346, 2.1425485 , 0.52638719, 0.15074759])
```

<!-- cell:23 type:code -->
```python
type(switchpoint), type(early_mean)
```
Output:
```
(pytensor.tensor.variable.TensorVariable,
 pytensor.tensor.variable.TensorVariable)
```

<!-- cell:24 type:markdown -->
Most importantly, anything distribution-like must have a `logp` method. This is what enables calculating the acceptance ratio for sampling:

<!-- cell:25 type:code -->
```python
try:
    switchpoint.logp({'switchpoint':55, 'early_mean_log__':1, 'late_mean_log__':1})
except (AttributeError, TypeError, KeyError):
    print('Model logp API has changed in modern pymc. Use model.point_logps() instead.')
```
Output:
```
Model logp API has changed in modern pymc. Use model.point_logps() instead.
```

<!-- cell:26 type:markdown -->
Ok, enough talk, lets sample:

<!-- cell:27 type:code -->
```python
with coaldis1:
    #stepper=pm.Metropolis()
    #idata = pm.sample(40000, step=stepper)
    idata = pm.sample(40000)
```
Output:
```
Multiprocess sampling (4 chains in 4 jobs)
```
Output:
```
CompoundStep
```
Output:
```
>NUTS: [early_mean, late_mean]
```
Output:
```
>Metropolis: [switchpoint]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 40_000 draw iterations (4_000 + 160_000 draws total) took 11 seconds.
```

<!-- cell:28 type:code -->
```python
az.summary(idata.sel(draw=slice(4000, None, 5)))
```
Output:
```
               mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
switchpoint  38.998  2.467  34.000   43.000      0.021    0.016   14206.0   15864.0    1.0
early_mean    3.067  0.285   2.549    3.610      0.002    0.001   25219.0   27077.0    1.0
late_mean     0.937  0.118   0.716    1.158      0.001    0.001   26498.0   26264.0    1.0
```

<!-- cell:29 type:code -->
```python
idata_burned = idata.sel(draw=slice(4000, None, 5))
az.plot_trace(idata_burned);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-20-output-1.png)

<!-- cell:30 type:markdown -->
A forestplot gives us 95% credible intervals...

<!-- cell:31 type:code -->
```python
az.plot_forest(idata_burned);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-21-output-1.png)

<!-- cell:32 type:code -->
```python
az.plot_autocorr(idata_burned);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-22-output-1.png)

<!-- cell:33 type:code -->
```python
plt.hist(idata.posterior['switchpoint'].values.flatten());
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-23-output-1.png)

<!-- cell:34 type:code -->
```python
idata_burned.posterior.to_dataframe().corr()
```
Output:
```
             switchpoint  early_mean  late_mean
switchpoint     1.000000   -0.265152  -0.244117
early_mean     -0.265152    1.000000   0.054666
late_mean      -0.244117    0.054666   1.000000
```

<!-- cell:35 type:markdown -->
## Imputation

Imputation of missing data vaues has a very nice process in Bayesian stats: just sample them from the posterior predictive. There is a very nice process to do this built into pync3..you could abuse this to calculate predictives at arbitrary points. (There is a better way for that, though, using Theano shared variables, so you might want to restrict this process to the situation where you need to impute a few values only).

Below we use -999 to handle mising data:

<!-- cell:36 type:code -->
```python
disasters_missing = np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
```

<!-- cell:37 type:code -->
```python
disasters_masked = np.ma.masked_values(disasters_missing, value=-999)
disasters_masked
```
Output:
```
masked_array(data=[4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3,
                   5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4,
                   2, 1, 3, --, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0,
                   3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
                   1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, --, 2, 1, 1, 1, 1, 2,
                   4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                   1, 0, 1],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False,  True,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False,  True, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False],
       fill_value=-999)
```

<!-- cell:38 type:code -->
```python
with pm.Model() as missing_data_model:
    switchpoint = pm.DiscreteUniform('switchpoint', lower=0, upper=len(disasters_masked))
    early_mean = pm.Exponential('early_mean', lam=1.)
    late_mean = pm.Exponential('late_mean', lam=1.)
    idx = np.arange(len(disasters_masked))
    rate = pm.Deterministic('rate', switch(switchpoint >= idx, early_mean, late_mean))
    disasters = pm.Poisson('disasters', rate, observed=disasters_masked)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/pymc/model/core.py:1316: ImputationWarning: Data in disasters contains missing values and will be automatically imputed from the sampling distribution.
  warnings.warn(impute_message, ImputationWarning)
```

<!-- cell:39 type:code -->
```python
try:
    pm.model_to_graphviz(missing_data_model)
except ImportError:
    print("graphviz not available, skipping model visualization")
```
Output:
```
graphviz not available, skipping model visualization
```

<!-- cell:40 type:markdown -->
By supplying a masked array to the likelihood part of our model, we ensure that the masked data points show up in our traces:

<!-- cell:41 type:code -->
```python
with missing_data_model:
    stepper=pm.Metropolis()
    idata_missing = pm.sample(40000, step=stepper)
```
Output:
```
Multiprocess sampling (4 chains in 4 jobs)
```
Output:
```
CompoundStep
```
Output:
```
>Metropolis: [switchpoint]
```
Output:
```
>Metropolis: [early_mean]
```
Output:
```
>Metropolis: [late_mean]
```
Output:
```
>Metropolis: [disasters_unobserved]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 40_000 draw iterations (4_000 + 160_000 draws total) took 13 seconds.
```

<!-- cell:42 type:code -->
```python
idata_m_burned = idata_missing.sel(draw=slice(4000, None, 5))
```

<!-- cell:43 type:code -->
```python
az.summary(idata_m_burned)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:596: RuntimeWarning: invalid value encountered in scalar divide
  (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:991: RuntimeWarning: invalid value encountered in scalar divide
  varsd = varvar / evar / 4
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:596: RuntimeWarning: invalid value encountered in scalar divide
  (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:991: RuntimeWarning: invalid value encountered in scalar divide
  varsd = varvar / evar / 4
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:596: RuntimeWarning: invalid value encountered in scalar divide
  (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/arviz/stats/diagnostics.py:991: RuntimeWarning: invalid value encountered in scalar divide
  varsd = varvar / evar / 4
```
Output:
```
                           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
switchpoint              38.817  2.437  35.000   43.000      0.027    0.017    8291.0   12237.0    1.0
disasters_unobserved[0]   2.160  1.795   0.000    5.000      0.017    0.010   10694.0   13068.0    1.0
disasters_unobserved[1]   0.935  0.982   0.000    3.000      0.007    0.006   17941.0   20618.0    1.0
early_mean                3.086  0.287   2.548    3.618      0.002    0.002   14854.0   16972.0    1.0
late_mean                 0.931  0.117   0.709    1.142      0.001    0.001   17153.0   18830.0    1.0
...                         ...    ...     ...      ...        ...      ...       ...       ...    ...
disasters[106]            0.000  0.000   0.000    0.000      0.000      NaN   28800.0   28800.0    NaN
disasters[107]            0.000  0.000   0.000    0.000      0.000      NaN   28800.0   28800.0    NaN
disasters[108]            1.000  0.000   1.000    1.000      0.000      NaN   28800.0   28800.0    NaN
disasters[109]            0.000  0.000   0.000    0.000      0.000      NaN   28800.0   28800.0    NaN
disasters[110]            1.000  0.000   1.000    1.000      0.000      NaN   28800.0   28800.0    NaN

[227 rows x 9 columns]
```

<!-- cell:44 type:code -->
```python
missing_data_model.free_RVs
```
Output:
```
[switchpoint, early_mean, late_mean, disasters_unobserved]
```

<!-- cell:45 type:code -->
```python
az.plot_trace(idata_m_burned);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-33-output-1.png)

<!-- cell:46 type:markdown -->
## Convergence of our model

Going back to the original model...

### Histograms every m samples

As a visual check, we plot histograms or kdeplots every 500 samples and check that they look identical.

<!-- cell:47 type:code -->
```python
import matplotlib.pyplot as plt

emtrace = idata_burned.posterior['early_mean'].values.flatten()
lmtrace = idata_burned.posterior['late_mean'].values.flatten()
smtrace = idata_burned.posterior['switchpoint'].values.flatten()

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes[0][0].hist(emtrace, bins=50)
axes[0][0].set_title("early_mean")
axes[0][1].hist(lmtrace, bins=50)
axes[0][1].set_title("late_mean")
axes[1][0].hist(smtrace, bins=50)
axes[1][0].set_title("switchpoint")
plt.tight_layout()
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-34-output-1.png)

<!-- cell:48 type:markdown -->
### Gewecke test

The gewecke test tests that the difference of means of chain-parts written as a Z-score oscilates between 1 and -1

$$\vert \mu_{\theta_1}  - \mu_{\theta_2}  \vert < 2 \sigma_{\theta_1 - \theta_2} $$



<!-- cell:49 type:code -->
```python
# Geweke diagnostic has been removed from modern pymc/arviz.
# Using az.rhat() for convergence assessment instead.
print('Geweke diagnostic deprecated; see Rhat diagnostics below.')
```
Output:
```
Geweke diagnostic deprecated; see Rhat diagnostics below.
```

<!-- cell:50 type:code -->
```python
# z-score output omitted (Geweke diagnostic deprecated)
```

<!-- cell:51 type:markdown -->
Here is a plot for `early_mean`. You sould really be plotting all of these...

<!-- cell:52 type:code -->
```python
# Geweke diagnostic is deprecated in modern pymc.
# See Rhat diagnostics below for convergence assessment.

```

<!-- cell:53 type:code -->
```python
# Geweke z-score plot omitted (diagnostic deprecated).
# Use az.plot_trace() and az.rhat() for convergence assessment.

```

<!-- cell:54 type:markdown -->
### Gelman-Rubin

For this test, which calculates 

$$\hat{R} = \sqrt{\frac{\hat{Var}(\theta)}{w}}$$

we need more than 1-chain. This is done through `njobs=4` (the defaukt is 2 and reported in `pm.summary`). See the trace below:

<!-- cell:55 type:code -->
```python
with coaldis1:
    stepper=pm.Metropolis()
    idata2 = pm.sample(40000, step=stepper, cores=4)
```
Output:
```
Multiprocess sampling (4 chains in 4 jobs)
```
Output:
```
CompoundStep
```
Output:
```
>Metropolis: [early_mean]
```
Output:
```
>Metropolis: [late_mean]
```
Output:
```
>Metropolis: [switchpoint]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 40_000 draw iterations (4_000 + 160_000 draws total) took 9 seconds.
```

<!-- cell:56 type:code -->
```python
idata2
```
Output:
```
Inference data with groups:
	> posterior
	> sample_stats
	> observed_data
```

<!-- cell:57 type:code -->
```python
idata2_cut = idata2.sel(draw=slice(4000, None, 5))
```

<!-- cell:58 type:code -->
```python
# gelman_rubin is now az.rhat()

az.rhat(idata2_cut)
```

<!-- cell:59 type:markdown -->
For the best results, each chain should be initialized to highly dispersed starting values for each stochastic node.

<!-- cell:60 type:markdown -->
A foresplot will show you the credible-interval consistency of our chains..

<!-- cell:61 type:code -->
```python
# forestplot is now az.plot_forest()

az.plot_forest(idata2_cut)
```
Output:
```
array([<Axes: title={'center': '94.0% HDI'}>], dtype=object)
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-43-output-2.png)

<!-- cell:62 type:markdown -->
### Autocorrelation

This can be probed by plotting the correlation plot and effective sample size

<!-- cell:63 type:code -->
```python
# effective_n is now az.ess()

az.ess(idata2_cut)
```

<!-- cell:64 type:code -->
```python
az.plot_autocorr(idata2_cut);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-45-output-1.png)

<!-- cell:65 type:code -->
```python
az.plot_autocorr(idata2);
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-46-output-1.png)

<!-- cell:66 type:markdown -->
## Posterior predictive checks

Finally let us peek into posterior predictive checks: something we'll talk more about soon.

<!-- cell:67 type:code -->
```python
with coaldis1:
    ppc = pm.sample_posterior_predictive(idata_burned)
```
Output:
```
Sampling: [disasters]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/TDMjbJ0KVXQ0cgT9PEjxe/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:68 type:code -->
```python
ppc.posterior_predictive['disasters'].shape
```
Output:
```
(4, 7200, 111)
```

<!-- cell:69 type:markdown -->
This gives us 200 samples at each of the 111 diasters we have data on.

<!-- cell:70 type:markdown -->
We plot the first 4 posteriors against actual data for consistency...

<!-- cell:71 type:code -->
```python
fig, axes = plt.subplots(1, 4, figsize=(12, 6))
print(axes.shape)
ppc_disasters = ppc.posterior_predictive['disasters'].values.reshape(-1, ppc.posterior_predictive['disasters'].shape[-1])
for obs, s, ax in zip(disasters_data, ppc_disasters.T, axes):
    print(obs)
    ax.hist(s, bins=10)
    ax.plot(obs+0.5, 1, 'ro')
```
Output:
```
(4,)
4
5
4
0
```
![Figure](https://rahuldave.com/posts/switchpoint/index_files/figure-html/cell-49-output-2.png)
