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
# L, epsilon, and other tweaking

##### Keywords: energy, hamiltonian monte carlo, NUTS, leapfrog, canonical distribution, microcanonical distribution, transition distribution, marginal energy distribution, data augmentation, classical mechanics, detailed balance, statistical mechanics, divergences, step-size, non-centered hierarchical model, hierarchical

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
sns.set_style("whitegrid")
sns.set_context("poster")
import pymc as pm
import arviz as az
```

<!-- cell:4 type:markdown -->
## Centered parametrization

We'll set up the modelled in what is called a "Centered" parametrization which tells us how $\theta_i$ is modelled: it is written to be directly dependent as a normal distribution from the hyper-parameters. 

<!-- cell:5 type:code -->
```python
J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])
```

<!-- cell:6 type:markdown -->
We set up our priors in a Hierarchical model to use this centered parametrization. We can say: the $\theta_j$ is drawn from a Normal hyper-prior distribution with parameters $\mu$ and $\tau$. Once we get a $\theta_j$ then can draw the means from it given the data $\sigma_j$ and one such draw corresponds to our data.

<!-- cell:7 type:markdown -->
$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\theta_{j} \sim \mathcal{N}(\mu, \tau)\\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

where $j \in \{1, \ldots, 8 \}$ and the
$\{ y_{j}, \sigma_{j} \}$ are given as data

<!-- cell:8 type:code -->
```python
with pm.Model() as schools1:

    mu = pm.Normal('mu', 0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
```

<!-- cell:9 type:code -->
```python
with schools1:
    idata1 = pm.sample(5000, cores=2, tune=1000, random_seed=42)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, theta]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 2 seconds.
```
Output:
```
There were 345 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details
```

<!-- cell:10 type:code -->
```python
az.rhat(idata1), az.ess(idata1)
```
Output:
```
(<xarray.Dataset> Size: 144B
 Dimensions:      (theta_dim_0: 8)
 Coordinates:
   * theta_dim_0  (theta_dim_0) int64 64B 0 1 2 3 4 5 6 7
 Data variables:
     mu           float64 8B 1.016
     theta        (theta_dim_0) float64 64B 1.004 1.029 1.019 ... 1.003 1.012
     tau          float64 8B 1.041
 Attributes:
     created_at:                 2026-03-06T17:10:48.645257+00:00
     arviz_version:              0.23.4
     inference_library:          pymc
     inference_library_version:  5.28.1
     sampling_time:              2.1188199520111084
     tuning_steps:               1000,
 <xarray.Dataset> Size: 144B
 Dimensions:      (theta_dim_0: 8)
 Coordinates:
   * theta_dim_0  (theta_dim_0) int64 64B 0 1 2 3 4 5 6 7
 Data variables:
     mu           float64 8B 1.333e+03
     theta        (theta_dim_0) float64 64B 1.231e+03 2.366e+03 ... 2.222e+03
     tau          float64 8B 51.55
 Attributes:
     created_at:                 2026-03-06T17:10:48.645257+00:00
     arviz_version:              0.23.4
     inference_library:          pymc
     inference_library_version:  5.28.1
     sampling_time:              2.1188199520111084
     tuning_steps:               1000)
```

<!-- cell:11 type:markdown -->
The Gelman-Rubin statistic seems fine, but notice how small the effective-n's are? Something is not quite right. Lets see traceplots.

<!-- cell:12 type:code -->
```python
az.plot_trace(idata1);
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-8-output-1.png)

<!-- cell:13 type:markdown -->
Its hard to pick the thetas out but $\tau$ looks not so white-noisy. Lets zoom in:

<!-- cell:14 type:code -->
```python
list(idata1.posterior.data_vars)
```
Output:
```
['mu', 'theta', 'tau']
```

<!-- cell:15 type:code -->
```python
az.plot_trace(idata1, var_names=['tau'])
```
Output:
```
array([[<Axes: title={'center': 'tau'}>, <Axes: title={'center': 'tau'}>]],
      dtype=object)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-10-output-2.png)

<!-- cell:16 type:markdown -->
There seems to be some stickiness at lower values in the trace. Zooming in even more helps us see this better:

<!-- cell:17 type:code -->
```python
logtau1 = np.log(idata1.posterior['tau'].values.flatten())
plt.plot(logtau1, alpha=0.6)
plt.axvline(5000, color="r")
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-11-output-1.png)

<!-- cell:18 type:markdown -->
### Tracking divergences

<!-- cell:19 type:code -->
```python
divergent = idata1.sample_stats['diverging'].values.flatten()
divergent
```
Output:
```
array([False, False, False, ..., False, False, False], shape=(10000,))
```

<!-- cell:20 type:code -->
```python
print('Number of Divergent %d' % divergent.nonzero()[0].size)
n_samples = len(divergent)
divperc = divergent.nonzero()[0].size / n_samples
print('Percentage of Divergent %.5f' % divperc)
```
Output:
```
Number of Divergent 345
Percentage of Divergent 0.03450
```

<!-- cell:21 type:code -->
```python
def biasplot(idata):
    logtau = np.log(idata.posterior['tau'].values.flatten())
    mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
    plt.figure(figsize=(8, 2))
    plt.axhline(0.7657852, lw=2.5, color='gray')
    plt.plot(mlogtau, lw=2.5)
    plt.ylim(0, 2)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of log(tau)')
    plt.title('MCMC estimation of cumsum log(tau)')
```

<!-- cell:22 type:code -->
```python
biasplot(idata1)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-15-output-1.png)

<!-- cell:23 type:code -->
```python
def funnelplot(idata):
    logtau = np.log(idata.posterior['tau'].values.flatten())
    divergent = idata.sample_stats['diverging'].values.flatten()
    theta0 = idata.posterior['theta'].sel(theta_dim_0=0).values.flatten()
    plt.figure(figsize=(5, 3))
    plt.scatter(theta0[divergent == 0], logtau[divergent == 0], s=10, color='r', alpha=0.1)
    plt.scatter(theta0[divergent == 1], logtau[divergent == 1], s=10, color='g')
    plt.axis([-20, 50, -6, 4])
    plt.ylabel('log(tau)')
    plt.xlabel('theta[0]')
    plt.title('scatter plot between log(tau) and theta[0]')
    plt.show()
```

<!-- cell:24 type:code -->
```python
funnelplot(idata1)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-17-output-1.png)

<!-- cell:25 type:markdown -->
You can also get an idea of your acceptance rate. 65% is decent for NUTS.

<!-- cell:26 type:code -->
```python
np.mean(idata1.sample_stats['acceptance_rate'].values)
```
Output:
```
np.float64(0.6640336761448085)
```

<!-- cell:27 type:markdown -->
### Where are the divergences coming from?

Divergences can be a sign of the symplectic integration going off to infinity, or a false positive. False positives occur because instead of waiting for infinity, some heuristics are used. This is typically true of divergences not deep in the funnel, where the curvature of the target distribution is high.





<!-- cell:28 type:markdown -->
## The effect of step-size

We can co-erce a smaller step-size $\epsilon$, and thus an ok symplectic integration from our sampler by increasing the target acceptance rate.

If we do this, then we have geometric ergodicity (we go everywhere!) between the Hamiltonian transitions (ie in the leapfrogs) and the target distribution. This should result in the divergence rate decreasing.

But if for some reason we do not have geometric ergodicity, then divergences will persist. This can happen deep in the funnel, where even drastic decreases in the step size are not able to explore the highly curved geometry.


<!-- cell:29 type:code -->
```python
with schools1:
    idata1_85 = pm.sample(5000, nuts={"target_accept": 0.85}, cores=2, tune=1000, random_seed=42)
with schools1:
    idata1_90 = pm.sample(5000, nuts={"target_accept": 0.90}, cores=2, tune=1000, random_seed=42)
with schools1:
    idata1_95 = pm.sample(5000, nuts={"target_accept": 0.95}, cores=2, tune=1000, random_seed=42)
with schools1:
    idata1_99 = pm.sample(5000, nuts={"target_accept": 0.99}, cores=2, tune=1000, random_seed=42)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, theta]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 3 seconds.
```
Output:
```
There were 258 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, theta]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 3 seconds.
```
Output:
```
There were 589 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, theta]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 6 seconds.
```
Output:
```
There were 55 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, theta]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 12 seconds.
```
Output:
```
There were 154 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```

<!-- cell:30 type:code -->
```python
for t in [idata1_85, idata1_90, idata1_95, idata1_99]:
    stats = t.sample_stats
    print("Acceptance", float(stats['acceptance_rate'].mean()),
          "Step Size", float(stats['step_size'].mean()),
          "Divergence", int(stats['diverging'].sum()))
```
Output:
```
Acceptance 0.7958025946365046 Step Size 0.1638554013031288 Divergence 258
Acceptance 0.743463369833306 Step Size 0.12258131417043726 Divergence 589
Acceptance 0.944499003562054 Step Size 0.058284299165064314 Divergence 55
Acceptance 0.9443072443822359 Step Size 0.049031843501610245 Divergence 154
```

<!-- cell:31 type:code -->
```python
for t in [idata1_85, idata1_90, idata1_95, idata1_99]:
    biasplot(t)
    funnelplot(t)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-1.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-2.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-3.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-4.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-5.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-6.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-7.png)
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-21-output-8.png)

<!-- cell:32 type:code -->
```python
plt.plot(np.log(idata1_99.posterior['tau'].values.flatten()))
```
Output:
```
[<matplotlib.lines.Line2D at 0x12cb94c20>]
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-22-output-1.png)

<!-- cell:33 type:code -->
```python
# Build a flat dataframe similar to pymc3's trace_to_dataframe
post = idata1_99.posterior
mu_vals = post['mu'].values.flatten()
tau_vals = post['tau'].values.flatten()
theta_vals = post['theta'].values.reshape(-1, J)  # (chains*draws, J)
df99 = pd.DataFrame({
    'mu': mu_vals,
    'tau': tau_vals,
    **{f'theta__{i}': theta_vals[:, i] for i in range(J)}
})
df99.head()
```
Output:
```
         mu       tau  theta__0  theta__1  theta__2  theta__3  theta__4  theta__5  theta__6  theta__7
0  4.811017  0.229226  4.554908  4.570202  4.919490  5.012854  4.784042  4.678617  4.932674  4.519733
1  2.163617  0.279277  2.095977  2.458981  1.969374  2.317489  2.531516  2.396885  1.882107  1.781204
2  2.402271  0.205453  2.422109  2.353038  2.441854  2.350568  2.268851  2.438293  2.770894  2.295364
3  2.415092  0.209596  2.089952  2.427947  2.560681  2.499267  2.377599  2.550576  2.402786  2.149137
4  1.608482  0.383412  1.639389  1.229288  1.666518  1.727573  2.090237  2.135949  1.731113  1.619115
```

<!-- cell:34 type:code -->
```python
df99.corr()
```
Output:
```
                mu       tau  theta__0  theta__1  theta__2  theta__3  theta__4  theta__5  theta__6  theta__7
mu        1.000000 -0.154059  0.457694  0.562359  0.577148  0.556217  0.603604  0.582580  0.449122  0.534485
tau      -0.154059  1.000000  0.365993  0.041698 -0.192284  0.000660 -0.273426 -0.183563  0.374843  0.018162
theta__0  0.457694  0.365993  1.000000  0.318444  0.228891  0.276287  0.202617  0.231239  0.401414  0.276843
theta__1  0.562359  0.041698  0.318444  1.000000  0.302810  0.352022  0.309377  0.320897  0.313889  0.311403
theta__2  0.577148 -0.192284  0.228891  0.302810  1.000000  0.331237  0.375720  0.378200  0.226457  0.297737
theta__3  0.556217  0.000660  0.276287  0.352022  0.331237  1.000000  0.338753  0.316212  0.302666  0.304513
theta__4  0.603604 -0.273426  0.202617  0.309377  0.375720  0.338753  1.000000  0.393377  0.203778  0.301566
theta__5  0.582580 -0.183563  0.231239  0.320897  0.378200  0.316212  0.393377  1.000000  0.226004  0.298863
theta__6  0.449122  0.374843  0.401414  0.313889  0.226457  0.302666  0.203778  0.226004  1.000000  0.274115
theta__7  0.534485  0.018162  0.276843  0.311403  0.297737  0.304513  0.301566  0.298863  0.274115  1.000000
```

<!-- cell:35 type:code -->
```python
idata1_99.sample_stats['diverging'].values.flatten().shape, df99.shape
```
Output:
```
((10000,), (10000, 10))
```

<!-- cell:36 type:code -->
```python
df99si = df99.reset_index()
df99si.head()
```
Output:
```
   index        mu       tau  theta__0  theta__1  theta__2  theta__3  theta__4  theta__5  theta__6  theta__7
0      0  4.811017  0.229226  4.554908  4.570202  4.919490  5.012854  4.784042  4.678617  4.932674  4.519733
1      1  2.163617  0.279277  2.095977  2.458981  1.969374  2.317489  2.531516  2.396885  1.882107  1.781204
2      2  2.402271  0.205453  2.422109  2.353038  2.441854  2.350568  2.268851  2.438293  2.770894  2.295364
3      3  2.415092  0.209596  2.089952  2.427947  2.560681  2.499267  2.377599  2.550576  2.402786  2.149137
4      4  1.608482  0.383412  1.639389  1.229288  1.666518  1.727573  2.090237  2.135949  1.731113  1.619115
```

<!-- cell:37 type:code -->
```python
diverging_99 = idata1_99.sample_stats['diverging'].values.flatten()
dvs = int(diverging_99.sum())
```

<!-- cell:38 type:code -->
```python
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates as pc
plt.figure(figsize=(16, 6))
orderedticks = ['index', 'mu', 'tau'] + ["theta__{}".format(i) for i in range(8)]
div_colors = list(islice(cycle(['g', 'g', 'g', 'g', 'g']), None, int(dvs)))
pc(df99si[diverging_99 == True][orderedticks], 'index', color=div_colors, alpha=0.1);
undiv_colors = list(islice(cycle(['k', 'k', 'k', 'k', 'k']), None, 200))
pc(df99si[diverging_99 == False].sample(200)[orderedticks], 'index', color=undiv_colors, alpha=0.05);
plt.gca().legend_.remove()
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-28-output-1.png)

<!-- cell:39 type:markdown -->
The divergences decrease, but dont totally go away, showing that we have lost some geometric ergodicity. And as we get to a very small step size we explore the funnel much better, but we are now taking our sampler more into a MH like random walk regime, and our sampler looks very strongly autocorrelated.

We know the fix, it is to move to a

<!-- cell:40 type:markdown -->
## Non-centered paramerization

We change our model to:

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\nu_{j} \sim \mathcal{N}(0, 1)\\
\theta_{j} = \mu + \tau\nu_j \\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

Notice how we have factored the dependency of $\theta$ on $\phi = \mu, \tau$ into a deterministic
transformation between the layers, leaving the
actively sampled variables uncorrelated. 

This does two things for us: it reduces steepness and curvature, making for better stepping. It also reduces the strong change in densities, and makes sampling from the transition distribution easier.

<!-- cell:41 type:code -->
```python
with pm.Model() as schools2:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Normal('nu', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * nu)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
```

<!-- cell:42 type:code -->
```python
with schools2:
    idata2 = pm.sample(5000, cores=2, tune=1000, random_seed=42)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, nu]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 1 seconds.
```
Output:
```
There were 7 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```

<!-- cell:43 type:markdown -->
And we reach the true value better as the number of samples increases, decreasing our bias

<!-- cell:44 type:code -->
```python
biasplot(idata2)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-31-output-1.png)

<!-- cell:45 type:markdown -->
How about our divergences? They have decreased too.

<!-- cell:46 type:code -->
```python
divergent = idata2.sample_stats['diverging'].values.flatten()
print('Number of Divergent %d' % divergent.nonzero()[0].size)
n_samples = len(divergent)
divperc = divergent.nonzero()[0].size / n_samples
print('Percentage of Divergent %.5f' % divperc)
```
Output:
```
Number of Divergent 7
Percentage of Divergent 0.00070
```

<!-- cell:47 type:code -->
```python
funnelplot(idata2)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-33-output-1.png)

<!-- cell:48 type:markdown -->
The divergences are infrequent and do not seem to concentrate anywhere, indicative of false positives. Lowering the step size should make them go away.

<!-- cell:49 type:markdown -->
### A smaller step size

<!-- cell:50 type:code -->
```python
with schools2:
    idata2_95 = pm.sample(5000, nuts={"target_accept": 0.95}, cores=2, tune=1000, random_seed=42)
```
Output:
```
Initializing NUTS using jitter+adapt_diag...
```
Output:
```
Multiprocess sampling (2 chains in 2 jobs)
```
Output:
```
NUTS: [mu, tau, nu]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 2 seconds.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```

<!-- cell:51 type:code -->
```python
biasplot(idata2_95)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-35-output-1.png)

<!-- cell:52 type:code -->
```python
funnelplot(idata2_95)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-36-output-1.png)

<!-- cell:53 type:markdown -->
Indeed at a smaller step-size our false-positive divergences go away, and the lower curvature in our parametrization ensures geometric ergodicity deep in our funnel

<!-- cell:54 type:markdown -->
## Path length L

If we choose too small a $L$ we are returning our HMC sampler to a random walk. How long must a leapfrog run explore a level set of the Hamiltonian (ie of the canonical distribution $p(p,q)$ beofre we force an accept-reject step and a momentum resample?

Clearly if we go too long we'll be coming back to the neighborhood of areas we might have reached in smaller trajectories. NUTS is one approach to adaptively fix this by not letting trajectories turn on themselves.

In the regular HMC sampler, for slightly complex problems, $L=100$ maybe a good place to start. For a fixed step-size $\epsilon$, we can now check the level of autocorrelation. If it is too much, we want a larger $L$.

Now, the problem with a fixed $L$ is that one $L$ does not work everywhere in a distribution. To see this, note that tails correspond to much higher energies. Thus the level-set surfaces are larger, and a fixed length $L$ trajectory only explores a small portion of this set before a momentum resampling takes us off. This is why a dynamic method like NUTS is a better choice.

<!-- cell:55 type:markdown -->
## Tuning HMC(NUTS)

This requires preliminary runs. In `pymc` some samples are dedicated to this, and an adaptive tuning is carried out according to algorithm 6 in the original NUTS paper: http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf .

But we have seem how to play with step-size within the context of the NUTS sampler, something which we might need to do for tetchy models. Clearly too large an $\epsilon$ can lead to loss of symplecticity. And too small a step-size will get us to a random walk for finite sized $L$. Clearly the adaptive approach in NUTS is a good idea.

In pymc's regular HMC sampler a knob `step_rand` is available which allows us a distribution to sample a step-size from. In the case that we are not adaptively tuning as in NUTS, allowing for this randomness allows for occasionally small values of $\epsilon$ even for large meaned distributions. Note that this should be done at the star of leapfrog, not in-between.

Another place where this is useful is where the exact solution to the Hamiltonian equations (gaussian distribution, harmonic oscillator hamiltonian resulting) has periodicity. If $L\epsilon$ is chosen to be $2\pi$ then our sampler will lack ergodicity. In such a case choose $\epsilon$ from a distribution (or for that matter, $L$).

Finally there are multiple ways to tweak the mass matrix. One might use a variational posterior to obtain a approximate covariance matrix for the target distribution $p(q)$. Or one could use the tuning samples for this purpose. But choosing the mass matrix as the inverse of the covariance matrix of the target is highly recommended, as it will maximally decorrelate parameters of the target distribution.

The covariance matrix also establishes a scale for each parameter. This scale can be used to tweak step size as well. Intuitively the variances are measures of curvature along a particular dimension, and choosing a stepsize in each parameter which accomodates this difference is likely to help symplecticity. I do not believe this optimization is available within pymc. This does not mean you are out of luck: you could simply redefine the parameters in a scaled form.

If you are combining HMC with other samplers, such as MH for discrete parameters in a gibbs based conditional scheme, then you might prefer smaller $L$ parameters to allow for the other parameters to be updated faster.


<!-- cell:56 type:markdown -->
## Efficiency of momentum resampling

When we talked about the most Hamiltonian-trajectory momentum resampling, we talked about its efficiency. The point there was that you want the marginal energy distribution to match the transition distribution induced by momentum resampling.

`pymc` gives us some handy stats to calculate this:

<!-- cell:57 type:code -->
```python
def resample_plot(idata):
    energy = idata.sample_stats['energy'].values.flatten()
    sns.histplot(energy - energy.mean(), label="P(E)", kde=True, stat="density")
    sns.histplot(np.diff(energy), label="p(E | q)", kde=True, stat="density")
    plt.legend();
    plt.xlabel("E - <E>")
```

<!-- cell:58 type:markdown -->
So let us see this for our original trace

<!-- cell:59 type:code -->
```python
resample_plot(idata1);
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-38-output-1.png)

<!-- cell:60 type:markdown -->
Awful. The momentum resamples here will do a very slow job of traversing this distribution. This is indicative of the second issue we were having with this centered model (the first was a large step size for the curvature causing loss of symplectivity): the momentum resampling simply cannot provide enough energy to traverse the large energy changes that occur in this hierarchical model.

<!-- cell:61 type:markdown -->
Note the caveat with such a plot obtained from our chains: it only tells us about the energies it explored: not the energies it ought to be exploring, as can be seen in the plot with `trace1_99` below. Still, a great diagnostic.

<!-- cell:62 type:code -->
```python
resample_plot(idata1_99)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-39-output-1.png)

<!-- cell:63 type:markdown -->
The match is much better for the non-centered version of our model.

<!-- cell:64 type:code -->
```python
resample_plot(idata2)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-40-output-1.png)

<!-- cell:65 type:code -->
```python
resample_plot(idata2_95)
```
![Figure](https://rahuldave.com/posts/hmctweaking/index_files/figure-html/cell-41-output-1.png)

<!-- cell:66 type:code -->
```python

```
