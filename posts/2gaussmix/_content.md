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

<!-- cell:2 type:markdown -->


<!-- cell:3 type:code -->
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
sns.set_style('whitegrid')
sns.set_context('poster')
import pymc as pm
import pytensor.tensor as pt
import arviz as az
```

<!-- cell:4 type:markdown -->
Here is a close set of 2 gaussians.

<!-- cell:5 type:code -->
```python
mu_true = np.array([-1, 1])
sigma_true = np.array([1, 1])
lambda_true = np.array([1/2, 1/2])
n = 100
from scipy.stats import multinomial
# Simulate from each distribution according to mixing proportion psi
z = multinomial.rvs(1, lambda_true, size=n)
data=np.array([np.random.normal(mu_true[i.astype('bool')][0], sigma_true[i.astype('bool')][0]) for i in z])
sns.histplot(data, bins=50, kde=True);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-4-output-1.png)

<!-- cell:6 type:markdown -->
We sample, without imposing any ordering.

<!-- cell:7 type:code -->
```python
with pm.Model() as model1:
    p = [1/2, 1/2]
    means = pm.Normal('means', mu=0, sigma=10, shape=2)
    points = pm.NormalMixture('obs', p, mu=means, sigma=1, observed=data)

```

<!-- cell:8 type:code -->
```python
with model1:
    trace1 = pm.sample(10000, tune=2000, target_accept=0.95)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (4 chains in 4 jobs)
```
Output:
```
NUTS: [means]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 4 seconds.
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details
```

<!-- cell:9 type:code -->
```python
az.plot_trace(trace1, combined=True);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-7-output-1.png)

<!-- cell:10 type:markdown -->
...and land up in a situation where we get mode-switching in one chain

<!-- cell:11 type:code -->
```python
az.plot_autocorr(trace1);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-8-output-1.png)

<!-- cell:12 type:code -->
```python
mtrace1 = trace1.posterior['means'].values.reshape(-1, 2)[::2]
mtrace1.shape
```
Output:
```
(20000, 2)
```

<!-- cell:13 type:code -->
```python
np.logspace(-10,2,13)
```
Output:
```
array([1.e-10, 1.e-09, 1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04, 1.e-03,
       1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
```

<!-- cell:14 type:markdown -->
As a result, the 2D posterior becomes multimodal..our sampler is having identifiability problems which show up in the ridiculously bad autocorrelation.

<!-- cell:15 type:code -->
```python
sns.kdeplot(x=mtrace1[:,0], y=mtrace1[:,1]);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-11-output-1.png)

<!-- cell:16 type:code -->
```python
az.plot_trace(trace1);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-12-output-1.png)

<!-- cell:17 type:markdown -->
We fix this by adding an ordering transform

<!-- cell:18 type:code -->
```python
with pm.Model() as model2:
    p = [1/2, 1/2]

    means = pm.Normal('means', mu=0, sigma=10, shape=2,
                  transform=pm.distributions.transforms.ordered,
                  initval=np.array([-1, 1]))
    points = pm.NormalMixture('obs', p, mu=means, sigma=1, observed=data)
```

<!-- cell:19 type:code -->
```python
with model2:
    trace2 = pm.sample(10000, tune=2000, target_accept=0.95)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (4 chains in 4 jobs)
```
Output:
```
NUTS: [means]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 6 seconds.
```

<!-- cell:20 type:code -->
```python
az.plot_trace(trace2, combined=True);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-15-output-1.png)

<!-- cell:21 type:markdown -->
...and the multi-modality goes away...

<!-- cell:22 type:code -->
```python
mtrace2 = trace2.posterior['means'].values.reshape(-1, 2)[::2]
mtrace2.shape
```
Output:
```
(20000, 2)
```

<!-- cell:23 type:code -->
```python
sns.kdeplot(x=mtrace2[:,0], y=mtrace2[:,1]);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-17-output-1.png)

<!-- cell:24 type:markdown -->
## ADVI

<!-- cell:25 type:code -->
```python
with model1:
    approx1 = pm.fit(n=15000, method="advi")
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 182.41
```

<!-- cell:26 type:code -->
```python
plt.plot(approx1.hist, '.-', alpha=0.2)
plt.ylim(150, 300)
```
Output:
```
(150.0, 300.0)
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-19-output-1.png)

<!-- cell:27 type:code -->
```python
samps1 = approx1.sample(10000)
```

<!-- cell:28 type:code -->
```python
az.plot_trace(samps1);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-21-output-1.png)

<!-- cell:29 type:code -->
```python
m = samps1.posterior['means'].values.reshape(-1, 2)
sns.kdeplot(x=m[:,0], y=m[:,1]);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-22-output-1.png)

<!-- cell:30 type:code -->
```python
with model2:
    approx2 = pm.fit(n=15000, method="advi")
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 178.72
```

<!-- cell:31 type:code -->
```python
plt.plot(approx2.hist, '.-', alpha=0.2)
plt.ylim(150, 300)
```
Output:
```
(150.0, 300.0)
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-24-output-1.png)

<!-- cell:32 type:code -->
```python
samps2 = approx2.sample(10000)
```

<!-- cell:33 type:code -->
```python
az.plot_trace(samps2);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-26-output-1.png)

<!-- cell:34 type:code -->
```python
m = samps2.posterior['means'].values.reshape(-1, 2)
sns.kdeplot(x=m[:,0], y=m[:,1]);
```
![Figure](https://rahuldave.com/posts/2gaussmix/index_files/figure-html/cell-27-output-1.png)
