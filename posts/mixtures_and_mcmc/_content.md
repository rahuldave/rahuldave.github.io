<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "graphviz",
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
We now do a study of learning mixture models with MCMC. We have already done this in the case of the Zero-Inflated Poisson Model, and will stick to Gaussian Mixture models for now.

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
## Mixture of 2 Gaussians, the old faithful data

We start by considering waiting times from the Old-Faithful Geyser at Yellowstone National Park.

<!-- cell:5 type:code -->
```python
ofdata=pd.read_csv("data/oldfaithful.csv")
ofdata.head()
```
Output:
```
   eruptions  waiting
0      3.600       79
1      1.800       54
2      3.333       74
3      2.283       62
4      4.533       85
```

<!-- cell:6 type:code -->
```python
sns.histplot(ofdata.waiting, kde=True);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-5-output-1.png)

<!-- cell:7 type:markdown -->
Visually, there seem to be two components to the waiting time, so let us model this using a mixture of two gaussians. Remember that this is a unsupervized model, and all we are doing is modelling $p(x)$ , with the assumption that there are two clusters and a hidden variable $z$ that indexes them.

Notice that these gaussians seem well separated. The separation of gaussians impacts how your sampler will perform.

<!-- cell:8 type:code -->
```python
with pm.Model() as ofmodel:
    p1 = pm.Uniform('p', 0, 1)
    p2 = 1 - p1
    p = pt.stack([p1, p2])
    assignment = pm.Categorical("assignment", p, 
                                shape=ofdata.shape[0])
    sds = pm.Uniform("sds", 0, 40, shape=2)
    centers = pm.Normal("centers", 
                        mu=np.array([50, 80]), 
                        sigma=np.array([20, 20]), 
                        shape=2)
    
    # and to combine it with the observations:
    observations = pm.Normal("obs", mu=centers[assignment], sigma=sds[assignment], observed=ofdata.waiting)
```

<!-- cell:9 type:code -->
```python
with ofmodel:
    oftrace = pm.sample(10000, target_accept=0.95)
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
>NUTS: [p, sds, centers]
```
Output:
```
>BinaryGibbsMetropolis: [assignment]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/dxNg8MqRfrVXkBgi9B0vB/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 77 seconds.
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/dxNg8MqRfrVXkBgi9B0vB/lib/python3.14/site-packages/arviz/stats/diagnostics.py:596: RuntimeWarning: invalid value encountered in scalar divide
  (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
```

<!-- cell:10 type:code -->
```python
pm.model_to_graphviz(ofmodel)
```

<!-- cell:11 type:code -->
```python
az.plot_trace(oftrace, var_names=["p", "centers", "sds"]);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-8-output-1.svg)

<!-- cell:12 type:code -->
```python
az.summary(oftrace, var_names=["p", "centers", "sds"])
```
Output:
```
              mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p            0.362  0.031   0.302    0.421      0.000    0.000   23423.0   27215.0    1.0
centers[0]  54.639  0.744  53.222   56.014      0.006    0.004   14494.0   19863.0    1.0
centers[1]  80.080  0.522  79.104   81.074      0.004    0.003   18198.0   21226.0    1.0
sds[0]       6.025  0.591   4.919    7.112      0.005    0.004   13170.0   17820.0    1.0
sds[1]       5.951  0.421   5.172    6.737      0.004    0.002   14442.0   20604.0    1.0
```

<!-- cell:13 type:code -->
```python
az.plot_autocorr(oftrace, var_names=['p', 'centers', 'sds']);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-9-output-1.png)

<!-- cell:14 type:code -->
```python
oftrace.posterior["centers"].mean(dim=["chain", "draw"]).values
```
Output:
```
array([54.63938979, 80.08034908])
```

<!-- cell:15 type:markdown -->
We can visualize the two clusters, suitably scales by the category-belonging probability by taking the posterior means. Note that this misses any smearing that might go into making the posterior predictive

<!-- cell:16 type:code -->
```python
from scipy.stats import norm
x = np.linspace(20, 120, 500)
# for pretty colors later in the book.
colors = ["#348ABD", "#A60628"] if oftrace.posterior["centers"].values[0, -1, 0] > oftrace.posterior["centers"].values[0, -1, 1] \
    else ["#A60628", "#348ABD"]

posterior_center_means = oftrace.posterior["centers"].mean(dim=["chain", "draw"]).values
posterior_std_means = oftrace.posterior["sds"].mean(dim=["chain", "draw"]).values
posterior_p_mean = oftrace.posterior["p"].mean(dim=["chain", "draw"]).values.item()

plt.hist(ofdata.waiting, bins=20, histtype="step", density=True, color="k",
     lw=2, label="histogram of data")
y = posterior_p_mean * norm.pdf(x, loc=posterior_center_means[0],
                                scale=posterior_std_means[0])
plt.plot(x, y, label="Cluster 0 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[1], alpha=0.3)

y = (1 - posterior_p_mean) * norm.pdf(x, loc=posterior_center_means[1],
                                      scale=posterior_std_means[1])
plt.plot(x, y, label="Cluster 1 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[0], alpha=0.3)

plt.legend(loc="upper left")
plt.title("Visualizing Clusters using posterior-mean parameters");
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-11-output-1.png)

<!-- cell:17 type:markdown -->
## A tetchy 3 Gaussian Model

Let us set up our data. Our analysis here follows that of https://colindcarroll.com/2018/07/20/why-im-excited-about-pymc3-v3.5.0/ , and we have chosen 3 gaussians reasonably close to each other to show the problems that arise!

<!-- cell:18 type:code -->
```python
mu_true = np.array([-2, 0, 2])
sigma_true = np.array([1, 1, 1])
lambda_true = np.array([1/3, 1/3, 1/3])
n = 100
from scipy.stats import multinomial
# Simulate from each distribution according to mixing proportion psi
z = multinomial.rvs(1, lambda_true, size=n)
data=np.array([np.random.normal(mu_true[i.astype('bool')][0], sigma_true[i.astype('bool')][0]) for i in z])
sns.histplot(data, bins=50, kde=True);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-13-output-1.png)

<!-- cell:19 type:code -->
```python
np.savetxt("data/3gv2.dat", data)
```

<!-- cell:20 type:code -->
```python
with pm.Model() as mof:
    #p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=3)
    p=[1/3, 1/3, 1/3]

    # cluster centers
    means = pm.Normal('means', mu=0, sigma=10, shape=3)


    #sds = pm.HalfCauchy('sds', 5, shape=3)
    sds = np.array([1., 1., 1.])
    
    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sigma=1., #sds[category],
                       observed=data)

```

<!-- cell:21 type:code -->
```python
with mof:
    trace_mof = pm.sample(10000)
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
>NUTS: [means]
```
Output:
```
>CategoricalGibbsMetropolis: [category]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/dxNg8MqRfrVXkBgi9B0vB/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 27 seconds.
```
Output:
```
There were 4 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```
Output:
```
The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details
```

<!-- cell:22 type:code -->
```python
az.plot_trace(trace_mof, var_names=["means"], combined=True);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-14-output-1.png)

<!-- cell:23 type:code -->
```python
az.plot_autocorr(trace_mof, var_names=['means']);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-18-output-1.png)

<!-- cell:24 type:markdown -->
## Problems with clusters and sampling

Some of the traces seem ok, but the autocorrelation is quite bad. And there is label-switching .This is because there are major problems with using MCMC for clustering.

AND THIS IS WITHOUT MODELING $p$ OR $\sigma$. It gets much worse otherwise! (it would be better if the gaussians were quite widely separated out).

These are firstly, the lack of parameter identifiability (the so called label-switching problem) and secondly, the multimodality of the posteriors.

We have seen non-identifiability before. Switching labels on the means and z's, for example, does not change the likelihoods. The problem with this is that cluster parameters cannot be compared across chains: what might be a cluster parameter in one chain could well belong to the other cluster in the second chain. Even within a single chain, indices might swap leading to a telltale back and forth in the traces for long chains or data not cleanly separated.

Also, the (joint) posteriors can be highly multimodal. One form of multimodality is the non-identifiability, though even without identifiability issues the posteriors are highly multimodal.

To quote the Stan manual:
>Bayesian inference fails in cases of high multimodality because there is no way to visit all of the modes in the posterior in appropriate proportions and thus no way to evaluate integrals involved in posterior predictive inference.
In light of these two problems, the advice often given in fitting clustering models is to try many different initializations and select the sample with the highest overall probability. It is also popular to use optimization-based point estimators such as expectation maximization or variational Bayes, which can be much more efficient than sampling-based approaches.

<!-- cell:25 type:markdown -->
### Some mitigation via ordering in pymc3

But this is not a panacea. Sampling is still very hard.


<!-- cell:26 type:code -->
```python
with pm.Model() as mof2:
    
    p = [1/3, 1/3, 1/3]

    # cluster centers
    means = pm.Normal('means', mu=0, sigma=10, shape=3,
                  transform=pm.distributions.transforms.ordered,
                  initval=np.array([-1, 0, 1]))


                                         
    # measurement error
    #sds = pm.Uniform('sds', lower=0, upper=20, shape=3)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sigma=1., #sds[category],
                       observed=data)

```

<!-- cell:27 type:code -->
```python
with mof2:
    trace_mof2 = pm.sample(10000)
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
>NUTS: [means]
```
Output:
```
>CategoricalGibbsMetropolis: [category]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/dxNg8MqRfrVXkBgi9B0vB/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 33 seconds.
```
Output:
```
There were 906 divergences after tuning. Increase `target_accept` or reparameterize.
```

<!-- cell:28 type:code -->
```python
az.plot_trace(trace_mof2, var_names=["means"], combined=True);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-19-output-1.png)

<!-- cell:29 type:code -->
```python
az.plot_autocorr(trace_mof2, var_names=['means']);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-22-output-1.png)

<!-- cell:30 type:markdown -->
## Full sampling is horrible, even with potentials

Now lets put Dirichlet based (and this is a strongly centering Dirichlet) prior on the probabilities

<!-- cell:31 type:code -->
```python
from scipy.stats import dirichlet
ds = dirichlet(alpha=[10,10,10]).rvs(1000)
```

<!-- cell:32 type:code -->
```python
"""
Visualize points on the 3-simplex (eg, the parameters of a
3-dimensional multinomial distributions) as a scatter plot 
contained within a 2D triangle.
David Andrzejewski (david.andrzej@gmail.com)
"""
import numpy as NP
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import matplotlib.patches as PA

def plotSimplex(points, fig=None, 
                vertexlabels=['1','2','3'],
                **kwargs):
    """
    Plot Nx3 points array on the 3-simplex 
    (with optionally labeled vertices) 
    
    kwargs will be passed along directly to matplotlib.pyplot.scatter    
    Returns Figure, caller must .show()
    """
    if(fig == None):        
        fig = P.figure()
    # Draw the triangle
    l1 = L.Line2D([0, 0.5, 1.0, 0], # xcoords
                  [0, NP.sqrt(3) / 2, 0, 0], # ycoords
                  color='k')
    fig.gca().add_line(l1)
    fig.gca().xaxis.set_major_locator(MT.NullLocator())
    fig.gca().yaxis.set_major_locator(MT.NullLocator())
    # Draw vertex labels
    fig.gca().text(-0.05, -0.05, vertexlabels[0])
    fig.gca().text(1.05, -0.05, vertexlabels[1])
    fig.gca().text(0.5, NP.sqrt(3) / 2 + 0.05, vertexlabels[2])
    # Project and draw the actual points
    projected = projectSimplex(points)
    P.scatter(projected[:,0], projected[:,1], **kwargs)              
    # Leave some buffer around the triangle for vertex labels
    fig.gca().set_xlim(-0.2, 1.2)
    fig.gca().set_ylim(-0.2, 1.2)

    return fig    

def projectSimplex(points):
    """ 
    Project probabilities on the 3-simplex to a 2D triangle
    
    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = NP.zeros((points.shape[0],2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * NP.sqrt(3))
        # Vector 1 - bisect out of lower left vertex 
        p1 = points[idx, 0]
        x = x - (1.0 / NP.sqrt(3)) * p1 * NP.cos(NP.pi / 6)
        y = y - (1.0 / NP.sqrt(3)) * p1 * NP.sin(NP.pi / 6)
        # Vector 2 - bisect out of lower right vertex  
        p2 = points[idx, 1]  
        x = x + (1.0 / NP.sqrt(3)) * p2 * NP.cos(NP.pi / 6)
        y = y - (1.0 / NP.sqrt(3)) * p2 * NP.sin(NP.pi / 6)        
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / NP.sqrt(3) * p3)
      
        tripts[idx,:] = (x,y)

    return tripts


```

<!-- cell:33 type:code -->
```python
plotSimplex(ds, s=20);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-23-output-1.png)

<!-- cell:34 type:markdown -->
The idea behind a `Potential` is something that is not part of the likelihood, but enforces a constraint by setting the probability to 0 if the constraint is violated. We use it here to give each cluster some membership and to order the means to remove the non-identifiability problem. See below for how its used.

The sampler below has a lot of problems. 

<!-- cell:35 type:code -->
```python
with pm.Model() as mofb:
    p = pm.Dirichlet('p', a=np.array([10., 10., 10.]), shape=3)
    # ensure all clusters have some points
    p_min_potential = pm.Potential('p_min_potential', pt.switch(pt.min(p) < .1, -np.inf, 0))
    # cluster centers
    means = pm.Normal('means', mu=0, sigma=10, shape=3, transform=pm.distributions.transforms.ordered,
                  initval=np.array([-1, 0, 1]))

    category = pm.Categorical('category',
                              p=p,
                              shape=data.shape[0])

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sigma=1., #sds[category],
                       observed=data)


```

<!-- cell:36 type:code -->
```python
with mofb:
    trace_mofb = pm.sample(10000, target_accept=0.95)
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
>NUTS: [p, means]
```
Output:
```
>CategoricalGibbsMetropolis: [category]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/dxNg8MqRfrVXkBgi9B0vB/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 60 seconds.
```
Output:
```
There were 33 divergences after tuning. Increase `target_accept` or reparameterize.
```

<!-- cell:37 type:code -->
```python
az.plot_trace(trace_mofb, var_names=["means", "p"], combined=True);
```
![Figure](https://rahuldave.com/posts/mixtures_and_mcmc/index_files/figure-html/cell-26-output-1.png)

<!-- cell:38 type:code -->
```python
az.summary(trace_mofb, var_names=["means", "p"])
```
Output:
```
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
means[0] -1.713  0.288  -2.268   -1.195      0.005    0.003    2970.0    6546.0    1.0
means[1] -0.248  0.435  -1.105    0.552      0.011    0.006    1694.0    2841.0    1.0
means[2]  1.967  0.320   1.373    2.562      0.005    0.002    4139.0   10717.0    1.0
p[0]      0.351  0.079   0.204    0.501      0.002    0.001    2244.0    5286.0    1.0
p[1]      0.355  0.078   0.215    0.507      0.001    0.001    2946.0    7071.0    1.0
p[2]      0.293  0.061   0.180    0.408      0.001    0.000    3595.0    8348.0    1.0
```

<!-- cell:39 type:markdown -->
### Making Problems go away

A lot will go away when identifiability improves through separated gaussians. But that changes the data. If we want any further improvement on this data, we are going to have to stop sampling so many discrete categoricals. And for that we will need a marginalization trick.
