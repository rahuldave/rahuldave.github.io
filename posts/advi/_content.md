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
import pymc as pm
import arviz as az
```

<!-- cell:3 type:markdown -->
## From CAVI to Stochastic CAVI to ADVI

One of the challenges of any posterior inference problem is the ability to scale. While VI is faster than the traditional MCMC, the CAVI algorithm described above fundamentally doesn't scale as it needs to run through the **entire dataset** each iteration. An alternative that is sometimes recommended is the Stochastic CAVI that uses gradient-based optimization. Using this approach, the algorithm only requires a subsample of the data set to iteratively optimize local and global parameters of the model. 

Stochastic CAVI is specifically used for conditionally conjugate models, but the ideas from it are applicable outside: the use of gradient (for gradient ascent) and the use of SGD style techniques: minibatch or fully stochastic.

Finally, we have seen how to implement SGD in Theano, and how pymc3 uses automatic differentiation under the hood to provide gradients for its NUTS sampler. This idea is used to replace CAVI with an automatically-calculated gradient-ascent algorithm, with stochastic updates that allow us to scale by not requiring the use of the complete dataset at each iteration.

<!-- cell:4 type:markdown -->
## ADVI in pymc: approximating a gaussian

<!-- cell:5 type:code -->
```python
data = np.random.randn(100)
```

<!-- cell:6 type:code -->
```python
with pm.Model() as model: 
    mu = pm.Normal('mu', mu=0, sigma=1)
    sd = pm.HalfNormal('sd', sigma=1)
    n = pm.Normal('n', mu=mu, sigma=sd, observed=data)
```

<!-- cell:7 type:code -->
```python
advifit = pm.ADVI( model=model)
```

<!-- cell:8 type:code -->
```python
advifit.fit(n=50000)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 141.86
```

<!-- cell:9 type:code -->
```python
elbo = -advifit.hist
```

<!-- cell:10 type:code -->
```python
plt.plot(elbo[::10]);
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-9-output-1.png)

<!-- cell:11 type:code -->
```python
# In modern pymc, shared_params is a list of pytensor shared variables
advifit.approx.params, type(advifit.approx.params)
```
Output:
```
([mu, rho], list)
```

<!-- cell:12 type:code -->
```python
advifit.approx.mean.eval(), advifit.approx.std.eval()
```
Output:
```
(array([ 0.09180503, -0.03707109]), array([0.10386893, 0.07518803]))
```

<!-- cell:13 type:code -->
```python
m = advifit.approx.mean.eval()[0]
s = advifit.approx.std.eval()[1]
m,s
```
Output:
```
(np.float64(0.09180503126952601), np.float64(0.07518803076110839))
```

<!-- cell:14 type:code -->
```python
sig = np.exp(advifit.approx.mean.eval()[1])
sig
```
Output:
```
np.float64(0.9636076266276797)
```

<!-- cell:15 type:code -->
```python
# approx.sample() returns InferenceData in modern pymc
trace = advifit.approx.sample(10000)
```

<!-- cell:16 type:code -->
```python
az.summary(trace)
```
Output:
```
arviz - WARNING - Shape validation failed: input_shape: (1, 10000), minimum_shape: (chains=2, draws=4)
```
Output:
```
     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu  0.091  0.104  -0.095    0.295      0.001    0.001   10404.0   10174.0    NaN
sd  0.967  0.073   0.835    1.105      0.001    0.001   10174.0    9918.0    NaN
```

<!-- cell:17 type:code -->
```python
az.plot_trace(trace);
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-16-output-1.png)

<!-- cell:18 type:code -->
```python
trace.posterior['mu'].values.mean(), trace.posterior['sd'].values.mean()
```
Output:
```
(np.float64(0.09119829924228555), np.float64(0.9665791981281656))
```

<!-- cell:19 type:code -->
```python
with model:
    idata_nuts = pm.sample(10000, tune=1000)
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
NUTS: [mu, sd]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 2 seconds.
```

<!-- cell:20 type:code -->
```python
az.summary(idata_nuts)
```
Output:
```
     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu  0.088  0.096  -0.094    0.269        0.0      0.0   39927.0   28782.0    1.0
sd  0.965  0.069   0.837    1.096        0.0      0.0   39103.0   28895.0    1.0
```

<!-- cell:21 type:code -->
```python
with model:
    pred = pm.sample_posterior_predictive(trace)
```
Output:
```
Sampling: [n]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:22 type:code -->
```python
pred.posterior_predictive['n'].shape
```
Output:
```
(1, 10000, 100)
```

<!-- cell:23 type:markdown -->
### Comparing the mu parameter

<!-- cell:24 type:code -->
```python
sns.kdeplot(idata_nuts.posterior['mu'].values.flatten(), label='NUTS')
sns.kdeplot(trace.posterior['mu'].values.flatten(), label='ADVI')
plt.legend();
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-22-output-1.png)

<!-- cell:25 type:code -->
```python
sns.kdeplot(idata_nuts.posterior['sd'].values.flatten(), label='NUTS')
sns.kdeplot(trace.posterior['sd'].values.flatten(), label='ADVI')
plt.legend();
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-23-output-1.png)

<!-- cell:26 type:markdown -->
### Comparing the data to the posterior-predictive

<!-- cell:27 type:code -->
```python
pred_n = pred.posterior_predictive['n'].values.reshape(-1, pred.posterior_predictive['n'].shape[-1])
pred_n[:,0].shape
```
Output:
```
(10000,)
```

<!-- cell:28 type:code -->
```python
sns.histplot(data, stat='density', kde=False, alpha=0.3)
sns.kdeplot(pred_n[:,0])
sns.kdeplot(pred_n[:,1])
sns.kdeplot(pred_n[:,50])
sns.kdeplot(pred_n[:,99])
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-25-output-1.png)

<!-- cell:29 type:markdown -->
## ADVI: what does it do?

Remember that in Variational inference, we decompose an aprroximate posterior in the mean-field approximation into a product of per-latent-variable posteriors. The approximate posterior is chosen from a pre-specified family of distributions to "variationally" minimize the KL-divergence (equivalently to maximize the ELBO) between itself and the true posterior.

$$ ELBO(q) = E_q[(log(p(z,x))] - E_q[log(q(z))] $$ 


This means that the ELBO must be painstakingly calculated and optimized with custom CAVI updates for each new model, and an approximating family chosen. If you choose to use a gradient based optimizer then you must supply gradients.

From the ADVI paper:

>ADVI solves this problem automatically. The user specifies the model, expressed as a program, and ADVI automatically generates a corresponding variational algorithm. The idea is to first automatically transform the inference problem into a common space and then to solve the variational optimization. Solving the problem in this common space solves variational inference for all models in a large class. 

Here is what ADVI does for us:

(1) The model undergoes transformations such that the latent parameters are transformed to representations where the 'new" parameters are unconstrained on the real-line. Specifically the joint $p(x, \theta)$ transforms to $p(x, \eta)$ where $\eta$ is unconstrained. We then define the approximating density $q$ and the posterior in terms of these transformed variable and minimize the KL-divergence between the transformed densities. This is done for *ALL* latent variables so that all of them are now defined on the same space. As a result we can use the same variational family for ALL parameters, and indeed for ALL models, as every parameter for every model is now defined on all of R. It should be clear from this that Discrete parameters must be marginalized out.

![ADVI transforms constrained latent variables (θ) to unconstrained real coordinate space (ζ) via invertible transformation T, enabling gradient-based optimization.](https://rahuldave.com/posts/advi/assets/TransformtoR.png)

Optimizing the KL-divergence implicitly assumes that the support of the approximating density lies within the support of the posterior. These transformations make sure that this is the case

(2) Ok, so now we must maximize our suitably transformed ELBO (the log full-data posterior will gain an additional term which is the determinant of the log of the Jacobian). Remember in variational inference that we are optimizing an expectation value with respect to the transformed approximate posterior. This posterior contains our transformed latent parameters so the gradient of this expectation is not simply defined.

What are we to do?

(3) We first choose as our family of approximating densities mean-field normal distributions. We'll tranform the always positive $\sigma$ params by simply taking their logs. 

The choice of Gaussians may make you think that we are doing a laplace (taylor series) approximation around the posterior mode, which is another method for approximate inference. This is not what we are doing here.

We still havent solved the problem of taking the gradient. Basically what we want to do is to push the gradient inside the expectation. For this, the distribution we use to calculate the expectation must be free of parameters we might compute the gradient with respect to.

So we indulge ourselves another transformation, which takes the approximate 1-D gaussian $q$ and standardizes it. The determinant of the jacobian of this transform is 1. This is the REPARAMETERIZATION TRICK and variants are available for other approximating families.

As a result of this, we can now compute the integral as a monte-carlo estimate over a standard Gaussian--superfast, and we can move the gradient inside the expectation (integral) to boot. This means that our job now becomes the calculation of the gradient of the full-data joint-distribution.

(4) We can replace full $x$ data by just one point (SGD) or mini-batch (some-$x$) and thus use noisy gradients to optimize the variational distribution. An
adaptively tuned step-size is used to provide good convergence.

<!-- cell:30 type:markdown -->
## Demonstrating ADVI in pymc

We wish to sample a 2D Posterior which looks something like below. Here the x and y axes are parameters.

<!-- cell:31 type:code -->
```python
cov=np.array([[1,0.8],[0.8,1]])
data = np.random.multivariate_normal([0,0], cov, size=1000)
sns.kdeplot(x=data[:,0], y=data[:,1], alpha=0.4);
plt.scatter(data[:,0], data[:,1], s=10, alpha=0.2)
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-26-output-1.png)

<!-- cell:32 type:code -->
```python
np.std(data[:,0]),np.std(data[:,1])
```
Output:
```
(np.float64(1.0291435394755577), np.float64(1.0564720867887827))
```

<!-- cell:33 type:markdown -->
Ok, so we just set up a simple sampler with no observed data

<!-- cell:34 type:code -->
```python
import pytensor.tensor as pt
cov=np.array([[0,0.8],[0.8,0]], dtype=np.float64)
with pm.Model() as mdensity:
    density = pm.MvNormal('density', mu=[0,0], cov=pt.fill_diagonal(cov,1), shape=2)
```

<!-- cell:35 type:markdown -->
We try and retrieve the posterior by sampling

<!-- cell:36 type:code -->
```python
with mdensity:
    mdtrace=pm.sample(10000)
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
NUTS: [density]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 3 seconds.
```

<!-- cell:37 type:code -->
```python
az.plot_trace(mdtrace);
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-30-output-1.png)

<!-- cell:38 type:markdown -->
We do a pretty good job:

<!-- cell:39 type:code -->
```python
density_vals = mdtrace.posterior['density'].values.reshape(-1, 2)
plt.scatter(density_vals[:,0], density_vals[:,1], s=5, alpha=0.1)
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-31-output-1.png)

<!-- cell:40 type:markdown -->
But when we sample using ADVI, the mean-field approximation means that we lose our correlation:

<!-- cell:41 type:code -->
```python
mdvar = pm.ADVI(model=mdensity)
mdvar.fit(n=40000)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 0.45251
```

<!-- cell:42 type:code -->
```python
plt.plot(-mdvar.hist[::10])
```
Output:
```
[<matplotlib.lines.Line2D at 0x116711550>]
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-33-output-1.png)

<!-- cell:43 type:code -->
```python
samps=mdvar.approx.sample(5000)
```

<!-- cell:44 type:code -->
```python
samps_vals = samps.posterior['density'].values.reshape(-1, 2)
plt.scatter(samps_vals[:,0], samps_vals[:,1], s=5, alpha=0.3)
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-35-output-1.png)

<!-- cell:45 type:markdown -->
A full rank fit also models the covariance parameters, and thus restores our correlation at the cost of more variational parameters to fit...

<!-- cell:46 type:code -->
```python
mdvar_fr = pm.FullRankADVI(model=mdensity)
mdvar_fr.fit(n=40000)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/WJgPh5nRFVZl0DU9tt8M7/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 0.033937
```

<!-- cell:47 type:code -->
```python
plt.plot(-mdvar_fr.hist[::10])
```
Output:
```
[<matplotlib.lines.Line2D at 0x1155b5400>]
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-37-output-1.png)

<!-- cell:48 type:code -->
```python
samps2=mdvar_fr.approx.sample(5000)
samps2_vals = samps2.posterior['density'].values.reshape(-1, 2)
plt.scatter(samps2_vals[:,0], samps2_vals[:,1], s=5, alpha=0.3)
```
![Figure](https://rahuldave.com/posts/advi/index_files/figure-html/cell-38-output-1.png)
