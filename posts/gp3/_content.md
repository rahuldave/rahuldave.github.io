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
#   "scikit-learn",
#   "scipy",
#   "seaborn",
# ]
# ///

```

<!-- cell:2 type:markdown -->
## Recall: Definition of Gaussian Process

<!-- cell:3 type:markdown -->
So lets assume we have this function vector 
$ f=(f(x_1),...f(x_n))$. If, for ANY choice of input points, $(x_1,...,x_n)$, the marginal distribution over f:

$$P(F) = \int_{f \not\in F} P(f) df$$

is multi-variate Gaussian, then the distribution $P(f)$ over the function f is said to be a Gaussian Process. 

We write a Gaussian Process thus:

$$f(x) \sim \mathcal{GP}(m(x), k(x,x\prime))$$

where the mean and covariance functions can be thought of as the infinite dimensional mean vector and covariance matrix respectively. 

By this we mean that the function $f(x)$ is drawn from a gaussian process with some probability. (It is worth repeating that a Gaussian Process defines a prior distribution over functions.) Once we have seen some data, this prior can be converted to a posterior over functions, thus restricting the set of functions that we can use based on the data. 

The consequence of the finite point marginal distribution over functions being multi-variate gaussian is that any $m$ observations in an arbitrary data set, $y = {y_1,...,y_n=m}$ can always be represented as a single point sampled from some $m$-variate Gaussian distribution. Thus, we can work backwards to 'partner' a GP with a data set, by marginalizing over the infinitely-many variables that we are not interested in, or have not observed. 

It is often assumed for simplicity that the mean of this 'partnered' GP is zero in every dimension. Then, the only thing that relates one observation to another is the covariance function  $k(x_i, x_j)$. 

There are many choices of covariance functions but a popular choice is the 'squared exponential',

$$ k(x_i, x_j) = \sigma_f^2 exp( \frac{-(x_i-x_j)^2}{2l^2}) $$

where $\sigma_f^2$ is the maximum allowable covariance (or amplitude) and $l$ is the distance in the independent variable between $x_i$ and $x_j$. Note that this is a function solely of the independent variable, rather than the dependent one i.e. the only thing that impacts the corvariance function is the distances between each set of points. 

Thus, it is the covariance function that dictates if small length scales will be more likely, and large ones will be less, for example, and thus that wiggly functions are more probable, or that more rather than less samples drawn from the GP will be wiggly.

<!-- cell:4 type:code -->
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
```

<!-- cell:5 type:code -->
```python
x_pred = np.linspace(0, 10, 1000)
```

<!-- cell:6 type:markdown -->
Calculate covariance based on the kernel

<!-- cell:7 type:code -->
```python
xold=np.arange(0,10,0.5)
xtrain, xtest = train_test_split(xold)
xtrain = np.sort(xtrain)
xtest = np.sort(xtest)
print(xtrain, xtest)
```
Output:
```
[0.  0.5 1.  1.5 2.  3.  4.  4.5 5.  6.  6.5 7.  7.5 8.5 9. ] [2.5 3.5 5.5 8.  9.5]
```

<!-- cell:8 type:markdown -->
At this point lets choose a regression function

<!-- cell:9 type:code -->
```python
"""The function to predict."""
def f(x):
    return x**.5*np.sin(x)
plt.plot(xold, f(xold), 'o');
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-6-output-1.png)

<!-- cell:10 type:code -->
```python
# Instantiate a Gaussian Process model
sigma_noise=0.4
noise = np.random.normal(0, sigma_noise, xtrain.shape[0])
ytrain = f(xtrain) + noise
```

<!-- cell:11 type:code -->
```python
plt.plot(xtrain, ytrain,'.')
plt.plot(x_pred, f(x_pred));
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-8-output-1.png)

<!-- cell:12 type:markdown -->
## Hyperparameter learning: Empirical Bayes or MCMC

<!-- cell:13 type:markdown -->
Above we very arbitrarily chose the parameters for the GP. In a bayesian context, these are parameters of our function prior, or they are hyperpriors. In analogy with mixtures, or hierarchical models, one way of obtaing the parameters would be to write out the joint distribution and do MCMC via a MH or Gibbs sampler. This is complex, but doable by setting priors on the amplitude and length scales of the kernel and the observational noise. 

The full MCMC approach can get expensive in the limit of many training points, (indeed the matrix inversion must be done at each gibbs step). Still that is better than nothing since the training size is the dimensionality the infinite-dimensional problem has been reduced to.

We do this MCMC using the marginaly likelihood, because, after all, we want to marginalize over our "infinite set" of functions. 

$$p(y|X) = \int_f p(y|f,X)p(f|X) df$$


We could also use type-2 maximum likelihood or empirical bayes, and maximize the marginal likelihood.

 
The Marginal likelihood given a GP prior and a gaussian likelihood is:
(you can obtain this from the properties of gaussians and their integrals)

$$\log p(y|X) = - \frac{n}{2}\log2\pi - \frac{1}{2}\log|K + \sigma^2I| - \frac{1}{2}y^T(K+\sigma^2I)^{-1}y $$

where K is the covariance matrix obtained from evaluating the kernel pairwise at allpoints of the training set $X$.

The  first term is a constant, the second  is a model complexity term, and the third term
is a quadratic form of the data. To understand the tradeoff between the data and complexity term, let us consider a squared exponential kernel
in 1 dimension.

Holding the amplitude parameter fixed, lets vary the length parameter. For short length scales, the covariance is very wiggly, and thus 1 only near the diagonal. On the other hand, for large length scales, reasonably separated points are not different, and the covariance is close to 1 throughout.

Thus for shorter length scales, the model complexity term is large (the determinant is a product of diagonal terms). The fit will be very good. For longer length scales, the model complexity term will be small, as the matrix will be all ones. The fit will be poor. This corresponds to our general understanding of bias and variance: a long length scale imposes a very unwiggly, line like model, which will underfit, where as a model with a short length scale will have many wiggles, and thus possibly overfit.

<!-- cell:14 type:markdown -->
To find the empirical bayes estimates of the hyperparameters, we will differentiate with respect to the hyperparameters, and set the derivatives to zero. Note that the noise variance can be added to the prior covariance hyperparameters, as is usual in the bayesian case.

Since the marginal likelihood is not convex, it can have local minima. 

Since this is a 'frequentist' optimization in a bayesian scenario, dont forget to crossvalidate or similar to get good parameter estimates.

Below we carry out the MCMC procedure using PyMC, and MLE for the marginal likelihood using `sklearn`.

<!-- cell:15 type:markdown -->
## Fitting a model using PyMC

At this point, you might be wondering how to do Bayesian inference in the case of GPs. After all, to get posteriors on the hyperparameters we have to marginalize over functions, or equivalently infinite parameters.

The answer is something you might have not seen until now, but something which as always an option if the marginal likelihood integrals are analytic. Instead of optimizing the marginal likelihood, simply set up the bayesian problem as a hyperparameter posterior estimation problem. And in GPs, the marginal likelihood is simply Gaussian.

PyMC lets us do that. See:

<!-- cell:16 type:code -->
```python
with pm.Model() as model:
    # priors on the covariance function hyperparameters
    #l = pm.Gamma('l', alpha=2, beta=1)
    l = pm.Uniform('l', 0, 10)
    # uninformative prior on the function variance
    s2_f = pm.HalfCauchy('s2_f', beta=10)
    # uninformative prior on the noise variance
    s2_n = pm.HalfCauchy('s2_n', beta=5)
    # covariance functions for the function f and the noise
    f_cov = s2_f**2 * pm.gp.cov.ExpQuad(1, l)
    mgp = pm.gp.Marginal(cov_func=f_cov)
    y_obs = mgp.marginal_likelihood('y_obs', X=xtrain.reshape(-1,1), y=ytrain, sigma=s2_n)
```

<!-- cell:17 type:code -->
```python
with model:
    marginal_post = pm.find_MAP()
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:18 type:code -->
```python
marginal_post
```
Output:
```
{'l_interval__': array(-1.43443575),
 's2_f_log__': array(-0.62292696),
 's2_n_log__': array(0.39578568),
 'l': array(1.92408483),
 's2_f': array(0.5363722),
 's2_n': array(1.4855509)}
```

<!-- cell:19 type:code -->
```python
with model:
    #step=pm.Metropolis()
    idata = pm.sample(10000, tune=2000, target_accept=0.85)
    #trace = pm.sample(10000, tune=2000, step=step)
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
NUTS: [l, s2_f, s2_n]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 11 seconds.
```
Output:
```
There were 6 divergences after tuning. Increase `target_accept` or reparameterize.
```

<!-- cell:20 type:code -->
```python
az.summary(idata)
```
Output:
```
       mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
l     1.488  0.485   0.704    2.201      0.005    0.025   13867.0   12910.0    1.0
s2_f  2.490  1.450   0.857    4.890      0.012    0.031   18560.0   15983.0    1.0
s2_n  0.498  0.195   0.226    0.843      0.002    0.004   16767.0   14743.0    1.0
```

<!-- cell:21 type:code -->
```python
az.plot_autocorr(idata);
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-14-output-1.png)

<!-- cell:22 type:code -->
```python
df = idata.posterior.to_dataframe().reset_index(drop=True)
df[['l', 's2_f', 's2_n']].corr()
```
Output:
```
             l      s2_f      s2_n
l     1.000000  0.510335  0.287065
s2_f  0.510335  1.000000 -0.026820
s2_n  0.287065 -0.026820  1.000000
```

<!-- cell:23 type:code -->
```python
az.plot_trace(idata, var_names=['l', 's2_f', 's2_n']);
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-16-output-1.png)

<!-- cell:24 type:code -->
```python
s2_f_samples = idata.posterior["s2_f"].values.flatten()
l_samples = idata.posterior["l"].values.flatten()
sns.kdeplot(x=s2_f_samples, y=l_samples)
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-17-output-1.png)

<!-- cell:25 type:markdown -->
We can get posterior predictive samples using `sample_posterior_predictive`, except that this being GPs, we will get back posterior predictive functions, not just parameter traces.

<!-- cell:26 type:code -->
```python
with model:
    fpred = mgp.conditional("fpred", Xnew=x_pred.reshape(-1,1), pred_noise=False)
    ypred = mgp.conditional("ypred", Xnew=x_pred.reshape(-1,1), pred_noise=True)
    gp_samples = pm.sample_posterior_predictive(idata, var_names=["fpred", "ypred"], random_seed=42)
```
Output:
```
Sampling: [fpred, ypred]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:27 type:code -->
```python
gp_samples.posterior_predictive["ypred"].shape
```
Output:
```
(4, 10000, 1000)
```

<!-- cell:28 type:code -->
```python
fpred_vals = gp_samples.posterior_predictive["fpred"].values.reshape(-1, x_pred.shape[0])
meanpred = fpred_vals.mean(axis=0)
meanpred.shape
```
Output:
```
(1000,)
```

<!-- cell:29 type:code -->
```python
ypred_vals = gp_samples.posterior_predictive["ypred"].values.reshape(-1, x_pred.shape[0])
ypred_vals[0].shape
```
Output:
```
(1000,)
```

<!-- cell:30 type:code -->
```python
with sns.plotting_context("poster"):
    [plt.plot(x_pred, y, color="gray", alpha=0.2) for y in fpred_vals[::5,:]]
    # overlay the observed data
    plt.plot(x_pred[::10], ypred_vals[123,::10], '.', color="green", label="noisy realization")
    plt.plot(xtrain, ytrain, 'ok', ms=10, label="train pts");
    plt.plot(x_pred, f(x_pred), 'r', ms=10, label="actual");
    plt.plot(x_pred, meanpred, 'b', ms=10, label="predicted");

    plt.xlabel("x");
    plt.ylabel("f(x)");
    plt.title("Posterior predictive distribution");
    plt.xlim(0,10);
    plt.legend();
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/IPython/core/events.py:96: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  func(*args, **kwargs)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/6dZLOYbEs6FIjQWRPDF00/lib/python3.14/site-packages/IPython/core/pylabtools.py:170: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  fig.canvas.print_figure(bytes_io, **kw)
```
![Figure](https://rahuldave.com/posts/gp3/index_files/figure-html/cell-22-output-2.png)

<!-- cell:31 type:markdown -->
## Where are GPs used?

- geostatistics with kriging, oil exploration
- spatial statistics
- as an interpolator (0 noise case) in weather simulations
- they are equivalent to many machine learning models such as kernelized regression, SVM and neural networks (some)
- ecology since model uncertainty is high
- they are the start of non-parametric regression
- time series analysis (see cover of BDA)
- because of the composability of kernels, in automatic statistical analysis (see the automatic statistician)
