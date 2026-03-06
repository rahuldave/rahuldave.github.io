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
# From the normal model to regression

##### Keywords: bayesian, normal-normal model, conjugate prior, MCMC engineering, pymc, regression

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
```

<!-- cell:4 type:markdown -->
The example we use here is described in McElreath's book, and our discussion mostly follows the one there, in sections 4.3 and 4.4. We have used code from https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3/blob/master/Chp_04.ipynb .

<!-- cell:5 type:markdown -->
## Howell's data

These are census data for the Dobe area !Kung San (https://en.wikipedia.org/wiki/%C7%83Kung_people). Nancy Howell conducted detailed quantitative studies of this Kalahari foraging population in the 1960s.

<!-- cell:6 type:code -->
```python
df = pd.read_csv('data/Howell1.csv', sep=';', header=0)
df.head()
```
Output:
```
    height     weight   age  male
0  151.765  47.825606  63.0     1
1  139.700  36.485807  63.0     0
2  136.525  31.864838  65.0     0
3  156.845  53.041914  41.0     1
4  145.415  41.276872  51.0     0
```

<!-- cell:7 type:code -->
```python
df.tail()
```
Output:
```
      height     weight   age  male
539  145.415  31.127751  17.0     1
540  162.560  52.163080  31.0     1
541  156.210  54.062497  21.0     0
542   71.120   8.051258   0.0     1
543  158.750  52.531624  68.0     1
```

<!-- cell:8 type:code -->
```python
plt.hist(df.height, bins=30);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-6-output-1.png)

<!-- cell:9 type:markdown -->
We get rid of the kids and only look at the heights of the adults.

<!-- cell:10 type:code -->
```python
df2 = df[df.age >= 18]
plt.hist(df2.height, bins=30);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-7-output-1.png)

<!-- cell:11 type:markdown -->
## Model for heights

We will now get relatively formal in specifying our models.

We will use a Normal model, $h \sim N(\mu, \sigma)$, and assume that the priors are independent. That is $p(\mu, \sigma) = p(\mu \vert \sigma) p(\sigma) = p(\mu)p(\sigma)$.

Our model is:

$$
h \sim N(\mu, \sigma)\\
\mu \sim Normal(148, 20)\\
\sigma \sim Unif(0, 50)
$$

<!-- cell:12 type:code -->
```python
import pymc as pm
import arviz as az
```

<!-- cell:13 type:markdown -->
### A pymc model

We now write the model as a pymc model. You will notice that the code pretty much follows our formal specification above.

When we were talking about gibbs in a Hierarchical model, we suggested that software uses the  Directed Acyclic Graph (DAG) structure of our models to make writing conditionals easy.

This is exactly what `pymc` does. A "Deterministic Random Variable" is one whose values are  determined by its parents, and a "Stochastic Random Variable" has these parental dependencies but is specified by them only upto some sampling. 

Deterministic nodes use `pm.Deterministic` or plain python code, while Stochastics come from distributions.

So for example, the likelihood node in the graph below,  depends on the mu and sigma nodes as its parents, but is not fully specified by them.

Specifically, a likelihood stochastic is an instance of an observed random variable.

A Stochastic always has a `logp`,  the log probability of the variables current value, given that of its parents. Clearly this is needed to do any metropolis stuff! `pymc` provides this for many distributions, but we can easily add in our own.

<!-- cell:14 type:code -->
```python
with pm.Model() as hm1:
    mu = pm.Normal('mu', mu=148, sigma=20)#parameter
    sigma = pm.Uniform('sigma', lower=0, upper=20)
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2.height)
```

<!-- cell:15 type:markdown -->
`initval` can be used to pass a starting point.

<!-- cell:16 type:code -->
```python
with hm1:
    #stepper=pm.Metropolis()
    idata_hm1 = pm.sample(10000, random_seed=42)# a start argument could be used here
    #as well
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
NUTS: [mu, sigma]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 3 seconds.
```

<!-- cell:17 type:code -->
```python
az.plot_trace(idata_hm1);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-11-output-1.png)

<!-- cell:18 type:code -->
```python
az.plot_autocorr(idata_hm1);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-12-output-1.png)

<!-- cell:19 type:code -->
```python
az.summary(idata_hm1)
```
Output:
```
          mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu     154.595  0.414  153.806  155.363      0.002    0.002   41524.0   29521.0    1.0
sigma    7.774  0.295    7.217    8.321      0.001    0.002   40984.0   28938.0    1.0
```

<!-- cell:20 type:markdown -->
A very nice hack to find the acceptance values is below, which I found at the totally worth reading tutorial [here](http://rlhick.people.wm.edu/stories/bayesian_7.html).

<!-- cell:21 type:code -->
```python
idata_hm1.posterior['mu'].values.flatten()[1:]
```
Output:
```
array([154.38822307, 154.6239876 , 154.09600048, ..., 155.4130613 ,
       154.05753314, 154.18202628], shape=(39999,))
```

<!-- cell:22 type:code -->
```python
idata_hm1.posterior['mu'].values.flatten()[:-1]
```
Output:
```
array([153.98630507, 154.38822307, 154.6239876 , ..., 154.67623946,
       155.4130613 , 154.05753314], shape=(39999,))
```

<!-- cell:23 type:code -->
```python
def acceptance(idata, paramname):
    vals = idata.posterior[paramname].values.flatten()
    accept = np.sum(vals[1:] != vals[:-1])
    return accept / vals.shape[0]
```

<!-- cell:24 type:code -->
```python
acceptance(idata_hm1, 'mu'), acceptance(idata_hm1, 'sigma')
```
Output:
```
(np.float64(0.9234), np.float64(0.9234))
```

<!-- cell:25 type:markdown -->
### How strong is the prior?

Above we had used a very diffuse value on the prior. But suppose we tamp it down instead, as in the model below.

<!-- cell:26 type:code -->
```python
with pm.Model() as hm1dumb:
    mu = pm.Normal('mu', mu=178, sigma=0.1)#parameter
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2.height)
```

<!-- cell:27 type:code -->
```python
with hm1dumb:
    #stepper=pm.Metropolis()
    idata_hm1dumb = pm.sample(10000, random_seed=42)
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
NUTS: [mu, sigma]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 2 seconds.
```

<!-- cell:28 type:code -->
```python
az.summary(idata_hm1dumb)
```
Output:
```
          mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu     177.864  0.100  177.674  178.050      0.001    0.000   39564.0   30585.0    1.0
sigma   24.607  0.936   22.863   26.372      0.005    0.005   39143.0   26123.0    1.0
```

<!-- cell:29 type:markdown -->
Ok, so our `mu` did not move much from our prior. But see how much larger our `sigma` became to compensate. One way to think about this is that . 0.1 standard deviation on the posterior corrsponds to a "prior N" of 100 points (1/0.1^2) in contrast to a 20 standard deviation. 

<!-- cell:30 type:markdown -->
## Regression: adding a predictor

<!-- cell:31 type:code -->
```python
plt.plot(df2.height, df2.weight, '.');
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-21-output-1.png)

<!-- cell:32 type:markdown -->
So lets write our model out now:

$$
h \sim N(\mu, \sigma)\\
\mu = intercept + slope \times weight\\
intercept \sim N(150, 100)\\
slope \sim N(0, 10)\\
\sigma \sim Unif(0, 50)
$$

Why should you not use a uniform prior on a slope?

<!-- cell:33 type:code -->
```python
with pm.Model() as hm2:
    intercept = pm.Normal('intercept', mu=150, sigma=100)
    slope = pm.Normal('slope', mu=0, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    # below is a deterministic
    mu = intercept + slope * df2.weight
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2.height)
    #stepper=pm.Metropolis()
    idata_hm2 = pm.sample(10000, random_seed=42)#, step=stepper)
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
NUTS: [intercept, slope, sigma]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 7 seconds.
```

<!-- cell:34 type:markdown -->
The $\mu$ now becomes a deterministic node, as it is fully known once we know the slope and intercept.

<!-- cell:35 type:code -->
```python
az.plot_trace(idata_hm2);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-23-output-1.png)

<!-- cell:36 type:code -->
```python
az.plot_autocorr(idata_hm2);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-24-output-1.png)

<!-- cell:37 type:code -->
```python
acceptance(idata_hm2, 'intercept'), acceptance(idata_hm2, 'slope'), acceptance(idata_hm2, 'sigma')
```
Output:
```
(np.float64(0.98395), np.float64(0.98395), np.float64(0.98395))
```

<!-- cell:38 type:markdown -->
Oops, what happened here? Our correlations are horrendous and the traces look awful. Our acceptance rates dont seem to be at fault.

<!-- cell:39 type:markdown -->
### Centering to remove correlation : identifying information in our parameters.

The slope and intercept are very highly correlated:

<!-- cell:40 type:code -->
```python
hm2df = idata_hm2.posterior[['intercept', 'slope', 'sigma']].to_dataframe().reset_index(drop=True)
hm2df.head()
```
Output:
```
    intercept     slope     sigma
0  114.298002  0.891360  5.432948
1  116.842475  0.846175  5.078828
2  117.525236  0.828611  5.166888
3  115.984489  0.859981  5.079850
4  112.704629  0.941411  5.197882
```

<!-- cell:41 type:code -->
```python
hm2df.corr()
```
Output:
```
           intercept     slope     sigma
intercept   1.000000 -0.989960  0.000104
slope      -0.989960  1.000000  0.000098
sigma       0.000104  0.000098  1.000000
```

<!-- cell:42 type:markdown -->
Indeed they are amost perfectly negatively correlated, the intercept compensating for the slope and vice-versa. This means that the two parameters carry the same information, and we have some kind of identifiability problem. We shall see more such problems as this course progresses.

We'll fix this here by centering our data. Lets subtract the mean of our weight variable.

<!-- cell:43 type:code -->
```python
with pm.Model() as hm2c:
    intercept = pm.Normal('intercept', mu=150, sigma=100)
    slope = pm.Normal('slope', mu=0, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    # below is a deterministic
    #mu = intercept + slope * (df2.weight -df2.weight.mean())
    mu = pm.Deterministic('mu', intercept + slope * (df2.weight -df2.weight.mean()))
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2.height)
    #stepper=pm.Metropolis()
    idata_hm2c = pm.sample(10000, random_seed=42)
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
NUTS: [intercept, slope, sigma]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 10_000 draw iterations (4_000 + 40_000 draws total) took 3 seconds.
```

<!-- cell:44 type:markdown -->
Notice we are now explicitly  modelling $\mu$ as a deterministic. This means that it will be added into our traceplots.

<!-- cell:45 type:code -->
```python
az.plot_trace(idata_hm2c);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-29-output-1.png)

<!-- cell:46 type:code -->
```python
az.plot_autocorr(idata_hm2c, var_names=['intercept', 'slope', 'sigma']);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-30-output-1.png)

<!-- cell:47 type:markdown -->
Everything is kosher now! What just happened?

The intercept  is now the expected value of the outcome when the predictor is at its mean. This means we have removed any dependence from the baseline value of the predictor.

<!-- cell:48 type:markdown -->
## Posteriors and Predictives

We can now plot the posterior means directly.  We take the traces on the $mu$s and find each ones mean, and plot them

<!-- cell:49 type:code -->
```python
# mu has shape (chains, draws, n_obs) — reshape to (chains*draws, n_obs)
mu_vals = idata_hm2c.posterior['mu'].values
mu_post = mu_vals.reshape(-1, mu_vals.shape[-1])
plt.plot(df2.weight, df2.height, 'o', label="data")
plt.plot(df2.weight, mu_post.mean(axis=0), label="posterior mean")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.legend();
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-31-output-1.png)

<!-- cell:50 type:markdown -->
However, by including the $\mu$ as a deterministic in our traces we only get to see the traces at existing data points. If we want the traces on a grid of weights, we'll have to explivitly plug in the intercept and slope traces in the regression formula

<!-- cell:51 type:code -->
```python
meanweight = df2.weight.mean()
weightgrid = np.arange(25, 71)
intercept_samples = idata_hm2c.posterior['intercept'].values.flatten()
slope_samples = idata_hm2c.posterior['slope'].values.flatten()
n_total = len(intercept_samples)
mu_pred = np.zeros((len(weightgrid), n_total))
for i, w in enumerate(weightgrid):
    mu_pred[i] = intercept_samples + slope_samples * (w - meanweight)
```

<!-- cell:52 type:markdown -->
We can see what the posterior density (on $\mu$) looks like at a given x (weight).

<!-- cell:53 type:code -->
```python
sns.kdeplot(mu_pred[30]);
plt.title("posterior density at weight {}".format(weightgrid[30]));
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-33-output-1.png)

<!-- cell:54 type:markdown -->
And we can create a plot of the posteriors using the HPD(Highest Posterior density interval) at each point on the grid.

<!-- cell:55 type:code -->
```python
mu_mean = mu_pred.mean(axis=1)
mu_hpd = az.hdi(mu_pred.T)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_93413/1897931254.py:2: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  mu_hpd = az.hdi(mu_pred.T)
```

<!-- cell:56 type:code -->
```python
plt.scatter(df2.weight, df2.height, c='b', alpha=0.3)
plt.plot(weightgrid, mu_mean, 'r')
plt.fill_between(weightgrid, mu_hpd[:,0], mu_hpd[:,1], color='r', alpha=0.5)
plt.xlabel('weight')
plt.ylabel('height')
plt.xlim([weightgrid[0], weightgrid[-1]]);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-35-output-1.png)

<!-- cell:57 type:markdown -->
Looks like our posterior $\mu$s are very tight. Why then is there such spread in the data?

<!-- cell:58 type:code -->
```python
hm2c.observed_RVs # these are the likelihoods
```
Output:
```
[height]
```

<!-- cell:59 type:markdown -->
### The posterior predictive

Remember that the traces for each $\mu \vert x$ are traces of the "deterministic" parameter $\mu$ at a given x. These are not traces of $y \vert x$s, or heights, but rather, traces of the expected-height at a given x.

Remember that we need to smear the posterior out with the sampling distribution to get the posterior predictive.

`pymc` makes this particularly simple for us, atleast at the points where we have data. We simply use the `pm.sample_posterior_predictive` function

<!-- cell:60 type:code -->
```python
with hm2c:
    postpred = pm.sample_posterior_predictive(idata_hm2c, random_seed=42)
```
Output:
```
Sampling: [height]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:61 type:markdown -->
Notice that these are at the 352 points where we have weights.

<!-- cell:62 type:code -->
```python
postpred.posterior_predictive['height'].shape
```
Output:
```
(4, 10000, 352)
```

<!-- cell:63 type:code -->
```python
# shape is (chains, draws, n_obs) — reshape to (chains*draws, n_obs)
pp_vals = postpred.posterior_predictive['height'].values
pp_height = pp_vals.reshape(-1, pp_vals.shape[-1])
postpred_means = pp_height.mean(axis=0)
```

<!-- cell:64 type:code -->
```python
postpred_hpd = az.hdi(pp_height)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_93413/2287744932.py:1: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  postpred_hpd = az.hdi(pp_height)
```

<!-- cell:65 type:markdown -->
Now when we plot the posterior predictives, we see that the error bars are much larger.

<!-- cell:66 type:code -->
```python
plt.plot(df2.weight, df2.height, '.', c='b', alpha=0.2)
plt.plot(weightgrid, mu_mean, 'r')
plt.fill_between(weightgrid, mu_hpd[:,0], mu_hpd[:,1], color='r', alpha=0.5)
yerr=[postpred_means - postpred_hpd[:,0], postpred_hpd[:,1] - postpred_means] 
plt.errorbar(df2.weight, postpred_means, yerr=yerr, fmt='--.', c='g', alpha=0.1, capthick=3)
plt.xlabel('weight')
plt.ylabel('height')
plt.xlim([weightgrid[0], weightgrid[-1]]);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-41-output-1.png)

<!-- cell:67 type:markdown -->
But we would like the posterior predictive at more than those 352 points...so here is the strategy we employ...

If we want 1000 samples (at each x) we, for each such sample, choose one of the posterior samples randomly, with replacement. We then get `gridsize` mus and one sigma from this posterior, which we then use to sample `gridsize` $y$'s from the likelihood.  This gives us 1000 $\times$ `gridsize` posterior predictives.

We have seen elsewhere how to do this using pytensor shared variables.

<!-- cell:68 type:code -->
```python
n_total
```
Output:
```
40000
```

<!-- cell:69 type:code -->
```python
n_ppredsamps=1000
weightgrid = np.arange(25, 71)
meanweight = df2.weight.mean()
sigma_samples = idata_hm2c.posterior['sigma'].values.flatten()
ppc_samples=np.zeros((len(weightgrid), n_ppredsamps))

for j in range(n_ppredsamps):
    k=np.random.randint(n_total)#samples with replacement
    musamps = intercept_samples[k] + slope_samples[k] * (weightgrid - meanweight)
    sigmasamp = sigma_samples[k]
    ppc_samples[:,j] = np.random.normal(musamps, sigmasamp)
```

<!-- cell:70 type:code -->
```python
ppc_samples_hpd = az.hdi(ppc_samples.T)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_93413/3695202599.py:1: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  ppc_samples_hpd = az.hdi(ppc_samples.T)
```

<!-- cell:71 type:markdown -->
And now we can plot using `fill_between`.

<!-- cell:72 type:code -->
```python
plt.scatter(df2.weight, df2.height, c='b', alpha=0.9)
plt.plot(weightgrid, mu_mean, 'r')
plt.fill_between(weightgrid, mu_hpd[:,0], mu_hpd[:,1], color='r', alpha=0.5)
plt.fill_between(weightgrid, ppc_samples_hpd[:,0], ppc_samples_hpd[:,1], color='green', alpha=0.2)


plt.xlabel('weight')
plt.ylabel('height')
plt.xlim([weightgrid[0], weightgrid[-1]]);
```
![Figure](https://rahuldave.com/posts/pymcnormalreg/index_files/figure-html/cell-45-output-1.png)

<!-- cell:73 type:code -->
```python
ppc_samples_hpd[-1], ppc_samples_hpd[22]
```
Output:
```
(array([168.2878227 , 187.24002009]), array([147.18010432, 166.12268716]))
```
