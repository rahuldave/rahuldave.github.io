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
We go back to our island tools data set to illustrate

- model comparison using WAIC
- model averaging using WAIC
- fighting overdispersion by making a hierarchical regression model.

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
import pytensor.tensor as pt
```

<!-- cell:4 type:code -->
```python
df=pd.read_csv("data/islands.csv", sep=';')
df
```
Output:
```
      culture  population contact  total_tools  mean_TU
0    Malekula        1100     low           13      3.2
1     Tikopia        1500     low           22      4.7
2  Santa Cruz        3600     low           24      4.0
3         Yap        4791    high           43      5.0
4    Lau Fiji        7400    high           33      5.0
5   Trobriand        8000    high           19      4.0
6       Chuuk        9200    high           40      3.8
7       Manus       13000     low           28      6.6
8       Tonga       17500    high           55      5.4
9      Hawaii      275000     low           71      6.6
```

<!-- cell:5 type:code -->
```python
df['logpop']=np.log(df.population)
df['clevel']=(df.contact=='high')*1
df
```
Output:
```
      culture  population contact  total_tools  mean_TU     logpop  clevel
0    Malekula        1100     low           13      3.2   7.003065       0
1     Tikopia        1500     low           22      4.7   7.313220       0
2  Santa Cruz        3600     low           24      4.0   8.188689       0
3         Yap        4791    high           43      5.0   8.474494       1
4    Lau Fiji        7400    high           33      5.0   8.909235       1
5   Trobriand        8000    high           19      4.0   8.987197       1
6       Chuuk        9200    high           40      3.8   9.126959       1
7       Manus       13000     low           28      6.6   9.472705       0
8       Tonga       17500    high           55      5.4   9.769956       1
9      Hawaii      275000     low           71      6.6  12.524526       0
```

<!-- cell:6 type:code -->
```python
def postscat(idata, thevars):
    d={}
    for v in thevars:
        d[v] = idata.posterior[v].values.flatten()
    df = pd.DataFrame.from_dict(d)
    g = sns.pairplot(df, diag_kind="kde", plot_kws={'s':10})
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    return g
```

<!-- cell:7 type:markdown -->
## Centered Model

As usual, centering the log-population fixes things:

<!-- cell:8 type:code -->
```python
df.logpop_c = df.logpop - df.logpop.mean()
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/630745569.py:1: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
  df.logpop_c = df.logpop - df.logpop.mean()
```

<!-- cell:9 type:code -->
```python
with pm.Model() as m1c:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel + betapc*df.clevel*df.logpop_c
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
```

<!-- cell:10 type:code -->
```python
with m1c:
    trace1c = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, betac, betapc, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 8 seconds.
```

<!-- cell:11 type:code -->
```python
az.plot_trace(trace1c);
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-10-output-1.png)

<!-- cell:12 type:code -->
```python
az.plot_autocorr(trace1c);
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-11-output-1.png)

<!-- cell:13 type:code -->
```python
az.ess(trace1c)
```

<!-- cell:14 type:code -->
```python
postscat(trace1c, ["betap", "betac", "betapc", "alpha"]);
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-13-output-1.png)

<!-- cell:15 type:code -->
```python
az.plot_posterior(trace1c);
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-14-output-1.png)

<!-- cell:16 type:markdown -->
## Model comparison for interaction significance

This is an example of feature selection, where we want to decide whether we should keep the interaction term or not, that is, whether the interaction is significant or not? We'll use model comparison to achieve this!

We can see some summary stats from this model:

<!-- cell:17 type:code -->
```python
dfsum=az.summary(trace1c)
dfsum
```
Output:
```
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
betap   0.263  0.035   0.196    0.329      0.000    0.000   13689.0   14213.0    1.0
betac   0.286  0.119   0.074    0.517      0.001    0.001   11563.0   12625.0    1.0
betapc  0.065  0.170  -0.249    0.395      0.001    0.001   15371.0   13990.0    1.0
alpha   3.311  0.091   3.143    3.483      0.001    0.001   11577.0   12893.0    1.0
```

<!-- cell:18 type:code -->
```python
# pm.dic is removed in modern pymc; use WAIC or LOO instead
az.waic(trace1c)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
Computed from 20000 posterior samples and 10 observations log-likelihood matrix.

          Estimate       SE
elpd_waic   -41.98     6.06
p_waic        6.98        -

There has been a warning during the calculation. Please check the results.
```

<!-- cell:19 type:code -->
```python
az.waic(trace1c)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
Computed from 20000 posterior samples and 10 observations log-likelihood matrix.

          Estimate       SE
elpd_waic   -41.98     6.06
p_waic        6.98        -

There has been a warning during the calculation. Please check the results.
```

<!-- cell:20 type:markdown -->
### Sampling from multiple different centered models

**(A)** Our complete model

**(B)** A model with no interaction

<!-- cell:21 type:code -->
```python
with pm.Model() as m2c_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2c_nopc = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, betac, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 6 seconds.
```

<!-- cell:22 type:markdown -->
**(C)** A model with no contact term

<!-- cell:23 type:code -->
```python
with pm.Model() as m2c_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2c_onlyp = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 5 seconds.
```

<!-- cell:24 type:markdown -->
**(D)** A model with only the contact term

<!-- cell:25 type:code -->
```python
with pm.Model() as m2c_onlyc:
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha +  betac*df.clevel
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2c_onlyc = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betac, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 6 seconds.
```

<!-- cell:26 type:markdown -->
**(E)** A model with only the intercept.

<!-- cell:27 type:code -->
```python
with pm.Model() as m2c_onlyic:
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2c_onlyic = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 3 seconds.
```

<!-- cell:28 type:markdown -->
We create a dictionary from these models and their traces, so that we can track the names as well

<!-- cell:29 type:code -->
```python
modeldict = {
    "m1c": {"idata": trace1c, "model": m1c},
    "m2c_nopc": {"idata": trace2c_nopc, "model": m2c_nopc},
    "m2c_onlyp": {"idata": trace2c_onlyp, "model": m2c_onlyp},
    "m2c_onlyc": {"idata": trace2c_onlyc, "model": m2c_onlyc},
    "m2c_onlyic": {"idata": trace2c_onlyic, "model": m2c_onlyic},
}
```

<!-- cell:30 type:code -->
```python
# Compute log_likelihood for each model so az.compare can use it
compare_dict = {}
for name, d in modeldict.items():
    idata = d["idata"]
    model = d["model"]
    if not hasattr(idata, "log_likelihood"):
        pm.compute_log_likelihood(idata, model=model)
    compare_dict[name] = idata
```

<!-- cell:31 type:markdown -->
## Comparing the models using WAIC

Finally we use `az.compare` to create a dataframe of comparisons.

<!-- cell:32 type:code -->
```python
comparedf = az.compare(compare_dict, ic="waic", method="pseudo-BMA")
comparedf.head()
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
            rank  elpd_waic     p_waic  elpd_diff        weight         se        dse  warning scale
m2c_nopc       0 -39.493347   4.195777   0.000000  8.694607e-01   5.528580   0.000000     True   log
m1c            1 -41.976041   6.978613   2.482694  7.261555e-02   6.064680   1.834442     True   log
m2c_onlyp      2 -42.202093   3.730087   2.708746  5.792372e-02   4.469968   3.961737     True   log
m2c_onlyic     3 -70.740050   8.273443  31.246703  2.338725e-14  15.812725  16.368695     True   log
m2c_onlyc      4 -75.175479  16.730025  35.682132  2.771385e-16  22.393098  22.251583     True   log
```

<!-- cell:33 type:code -->
```python
# comparedf already has model names as index from az.compare
comparedf
```
Output:
```
            rank  elpd_waic     p_waic  elpd_diff        weight         se        dse  warning scale
m2c_nopc       0 -39.493347   4.195777   0.000000  8.694607e-01   5.528580   0.000000     True   log
m1c            1 -41.976041   6.978613   2.482694  7.261555e-02   6.064680   1.834442     True   log
m2c_onlyp      2 -42.202093   3.730087   2.708746  5.792372e-02   4.469968   3.961737     True   log
m2c_onlyic     3 -70.740050   8.273443  31.246703  2.338725e-14  15.812725  16.368695     True   log
m2c_onlyc      4 -75.175479  16.730025  35.682132  2.771385e-16  22.393098  22.251583     True   log
```

<!-- cell:34 type:markdown -->
From McElreath, here is how to read this table:

>(1)	WAIC is obviously WAIC for each model. Smaller WAIC indicates better estimated out-of-sample deviance.

>(2)	pWAIC is the estimated effective number of parameters. This provides a clue as to how flexible each model is in fitting the sample.

>(3)	dWAIC is the difference between each WAIC and the lowest WAIC. Since only relative deviance matters, this column shows the differences in relative fashion.

>(4)	weight is the AKAIKE WEIGHT for each model. These values are transformed information criterion values. I'll explain them below.

>(5)	SE is the standard error of the WAIC estimate. WAIC is an estimate, and provided the sample size N is large enough, its uncertainty will be well approximated by its standard error. So this SE value isn't necessarily very precise, but it does provide a check against overconfidence in differences between WAIC values.

>(6)	dSE is the standard error of the difference in WAIC between each model and the top-ranked model. So it is missing for the top model. 

>The weight for a model i in a set of m models is given by:

$$w_i = \frac{exp(-\frac{1}{2}dWAIC_i)}{\sum_j exp(-\frac{1}{2}dWAIC_j)}$$

>The Akaike weight formula might look rather odd, but really all it is doing is putting WAIC on a probability scale, so it just undoes the multiplication by −2 and then exponentiates to reverse the log transformation. Then it standardizes by dividing by the total. So each weight will be a number from 0 to 1, and the weights together always sum to 1. Now larger values are better.

>But what do these weights mean? 

>Akaike's interpretation:

>A model's weight is an estimate of the probability that the model will make the best predictions on new data, conditional on the set of models considered...the Akaike weights are analogous to posterior probabilities of models, conditional on expected future data.

>So you can heuristically read each weight as an estimated probability that each model will perform best on future data. In simulation at least, interpreting weights in this way turns out to be appropriate. (McElreath 199-200)

<!-- cell:35 type:markdown -->
We can make visual comparison plots in the style of McElreath's book. We can see that all the weight is in the no-interaction, full, and only log(population) models.

<!-- cell:36 type:code -->
```python
az.plot_compare(comparedf)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-26-output-1.png)

<!-- cell:37 type:markdown -->
### Comparing for non-centered models

We can redo the coparison for non-centered models

<!-- cell:38 type:code -->
```python
with pm.Model() as m1:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel + betapc*df.clevel*df.logpop
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace1 = pm.sample(10000, tune=2000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, betac, betapc, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 17 seconds.
```

<!-- cell:39 type:code -->
```python
with pm.Model() as m2_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2_onlyp = pm.sample(10000, tune=2000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 6 seconds.
```

<!-- cell:40 type:code -->
```python
with pm.Model() as m2_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
    trace2_nopc = pm.sample(10000, tune=2000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, betac, alpha]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 2_000 tune and 10_000 draw iterations (8_000 + 40_000 draws total) took 9 seconds.
```

<!-- cell:41 type:code -->
```python
modeldict2 = {
    "m1": {"idata": trace1, "model": m1},
    "m2_nopc": {"idata": trace2_nopc, "model": m2_nopc},
    "m2_onlyp": {"idata": trace2_onlyp, "model": m2_onlyp},
    "m2_onlyc": {"idata": trace2c_onlyc, "model": m2c_onlyc},
    "m2_onlyic": {"idata": trace2c_onlyic, "model": m2c_onlyic},
}
```

<!-- cell:42 type:code -->
```python
compare_dict2 = {}
for name, d in modeldict2.items():
    idata = d["idata"]
    model = d["model"]
    if not hasattr(idata, "log_likelihood"):
        pm.compute_log_likelihood(idata, model=model)
    compare_dict2[name] = idata
```

<!-- cell:43 type:code -->
```python
comparedf2 = az.compare(compare_dict2, ic="waic", method="pseudo-BMA")
comparedf2
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
           rank  elpd_waic     p_waic  elpd_diff        weight         se        dse  warning scale
m2_nopc       0 -39.574344   4.268525   0.000000  6.002391e-01   5.533746   0.000000     True   log
m1            1 -40.096776   4.906833   0.522432  3.559876e-01   5.627265   0.579995     True   log
m2_onlyp      2 -42.192647   3.725374   2.618303  4.377334e-02   4.460087   3.965167     True   log
m2_onlyic     3 -70.740050   8.273443  31.165706  1.750773e-14  15.812725  16.357640     True   log
m2_onlyc      4 -75.175479  16.730025  35.601135  2.074663e-16  22.393098  22.238972     True   log
```

<!-- cell:44 type:markdown -->
What we find now is that the full-model has much more weight.

<!-- cell:45 type:code -->
```python
az.plot_compare(comparedf2)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-33-output-1.png)

<!-- cell:46 type:markdown -->
In either the centered or non-centered case, our top model excludes the interaction, but the second top model includes it. In the centered case, the non-interacting model has most of the weight, while in the non-centered model, the weights were more equally shared.

In a situation where the interaction model has so much weight, we can say its probably overfit. So in a sense, centering even helps us with our overfitting issues by clearly preferring the sans-interaction model, as it removes correlation and thus spurious weight being borrowed.

<!-- cell:47 type:markdown -->
## Computing the (counterfactual) posterior predictive for checking

We now write some code to compute the posterior predictive at arbitrary points without having to use pytensor shared variables and sample_posterior_predictive, in two different counterfactual situations of low contact and high contact. Since some of our models omit certain terms, we use traces with 0s in them to construct a general function to do this.

<!-- cell:48 type:code -->
```python
def trace_or_zero(idata, name):
    if name in idata.posterior:
        return idata.posterior[name].values.flatten()
    else:
        nsamples = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
        return np.zeros(nsamples)
```

<!-- cell:49 type:code -->
```python
# Number of total samples = chains * draws
nsamples = trace1c.posterior.sizes["chain"] * trace1c.posterior.sizes["draw"]
nsamples, trace1c.posterior['alpha'].values.flatten().shape[0]
```
Output:
```
(20000, 20000)
```

<!-- cell:50 type:code -->
```python
from scipy.stats import poisson
def compute_pp(lpgrid, idata, contact=0):
    alphatrace = trace_or_zero(idata, 'alpha')
    betaptrace = trace_or_zero(idata, 'betap')
    betactrace = trace_or_zero(idata, 'betac')
    betapctrace = trace_or_zero(idata, 'betapc')
    tl = len(alphatrace)
    gl = lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid_c = lpgrid - lpgrid.mean()
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid_c[i] + betactrace*contact + betapctrace*contact*lpgrid_c[i]
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```

<!-- cell:51 type:markdown -->
We compute the posterior predictive in the counterfactual cases: remember what we are doing here is turning on and off a feature.

<!-- cell:52 type:code -->
```python
lpgrid = np.linspace(6,13,30)
pplow = compute_pp(lpgrid, trace1c)
pphigh = compute_pp(lpgrid, trace1c, contact=1)
```

<!-- cell:53 type:markdown -->
We compute the medians and the hpds, and plot these against the data

<!-- cell:54 type:code -->
```python
pplowmed = np.median(pplow, axis=1)
pplowhpd = az.hdi(pplow.T)
pphighmed = np.median(pphigh, axis=1)
pphighhpd = az.hdi(pphigh.T)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/375403103.py:2: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pplowhpd = az.hdi(pplow.T)
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/375403103.py:4: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pphighhpd = az.hdi(pphigh.T)
```

<!-- cell:55 type:code -->
```python
with sns.plotting_context('poster'):
    plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'.', color="g")
    plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'.', color="r")
    plt.plot(lpgrid, pphighmed, color="g", label="c=1")
    plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
    plt.plot(lpgrid, pplowmed, color="r", label="c=0")
    plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-39-output-1.png)

<!-- cell:56 type:markdown -->
This is for the full centered model. The high contact predictive and data is in green. We undertake this exercise as a prelude to ensembling the models with high Akaike weights

<!-- cell:57 type:markdown -->
## Ensembling

Ensembles are a good way to combine models where one model may be good at something and the other at something else. Ensembles also help with overfitting if the variance cancels out between the ensemble members: they would all probably overfit in slightly different ways. Lets write a function to do our ensembling for us.

<!-- cell:58 type:code -->
```python
def ensemble(grid, modeldict, comparedf, modelnames, contact=0):
    accum_pp=0
    accum_weight=0
    for m in modelnames:
        weight = comparedf.loc[m]['weight']
        pp = compute_pp(grid, modeldict[m]["idata"], contact)
        print(m, weight, np.median(pp))
        accum_pp += pp*weight
        accum_weight +=weight
    return accum_pp/accum_weight
```

<!-- cell:59 type:code -->
```python
ens_pp_low = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'])
```
Output:
```
m1c 0.07261554626595455 28.0
m2c_nopc 0.8694607349682668 28.0
m2c_onlyp 0.057923718765755104 33.0
```

<!-- cell:60 type:code -->
```python
ens_pp_high = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'], contact=1)
```
Output:
```
m1c 0.07261554626595455 37.0
m2c_nopc 0.8694607349682668 37.0
```
Output:
```
m2c_onlyp 0.057923718765755104 32.0
```

<!-- cell:61 type:code -->
```python
with sns.plotting_context('poster'):
    pplowmed = np.median(ens_pp_low, axis=1)
    pplowhpd = az.hdi(ens_pp_low.T)
    pphighmed = np.median(ens_pp_high, axis=1)
    pphighhpd = az.hdi(ens_pp_high.T)
    plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
    plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
    plt.plot(lpgrid, pphighmed, color="g")
    plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
    plt.plot(lpgrid, pplowmed, color="r")
    plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/628422166.py:3: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pplowhpd = az.hdi(ens_pp_low.T)
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/628422166.py:5: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pphighhpd = az.hdi(ens_pp_high.T)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-43-output-2.png)

<!-- cell:62 type:markdown -->
The ensemble gives sensible limits and even regularizes down the green band at high population by giving more weight to the no-interaction model.

<!-- cell:63 type:markdown -->
## Hierarchical Modelling

**Overdispersion** is a problem one finds in most poisson models where the variance of the data is larger than the mean, which is the constraint the poisson distribution imposes.

To simplify things, let us consider here, only the model with log(population). Since there is no contact variable, there are no counterfactual plots and we can view the posterior predictive.

<!-- cell:64 type:code -->
```python
ppsamps = compute_pp(lpgrid, trace2c_onlyp)
ppmed = np.median(ppsamps, axis=1)
pphpd = az.hdi(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
#plt.ylim([0, 300])
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/478959609.py:3: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pphpd = az.hdi(ppsamps.T)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-44-output-2.png)

<!-- cell:65 type:markdown -->
By taking the ratio of the posterior-predictive variance to the posterior-predictive mean, we see that the model is overdispersed.

<!-- cell:66 type:code -->
```python
ppvar=np.var(ppsamps, axis=1)
ppmean=np.mean(ppsamps, axis=1)
```

<!-- cell:67 type:code -->
```python
ppvar/ppmean
```
Output:
```
array([1.28609537, 1.26022013, 1.25911084, 1.22957116, 1.21939143,
       1.21164235, 1.20977116, 1.17637416, 1.1547616 , 1.15159053,
       1.14548628, 1.11847757, 1.10974092, 1.11537954, 1.09401532,
       1.11399381, 1.12332672, 1.13285241, 1.11783819, 1.13130799,
       1.15628753, 1.19329506, 1.22502034, 1.2738189 , 1.31209513,
       1.39818014, 1.45712839, 1.56125456, 1.68234374, 1.80818061])
```

<!-- cell:68 type:markdown -->
Overdispersion can be fixed by considering a mixture model. We shall see this next week. But hierarchical modelling is also a great way to do this.

### Varying Intercepts hierarchical model

What we are basically doing is splitting the intercept into a value constant across the societies and a residual which is society dependent. It is this residual that we will assume is drawn from a gaussian with 0 mean and `sigmasoc` ($\sigma_{society}$) standard deviation. Since there is a varying intercept for **every** observation, $\sigma_{society}$ lands up as an estimate of overdispersion amongst societies.

<!-- cell:69 type:code -->
```python
with pm.Model() as m3c:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    sigmasoc = pm.HalfCauchy("sigmasoc", 1)
    alphasoc = pm.Normal("alphasoc", 0, sigmasoc, shape=df.shape[0])
    loglam = alpha + alphasoc + betap*df.logpop_c 
    y = pm.Poisson("ntools", mu=pt.exp(loglam), observed=df.total_tools)
```

<!-- cell:70 type:code -->
```python
with m3c:
    trace3 = pm.sample(5000, tune=1000, idata_kwargs={"log_likelihood": True})
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
NUTS: [betap, alpha, sigmasoc, alphasoc]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 4 seconds.
```

<!-- cell:71 type:markdown -->
Notice that we are fitting 13 parameters to 10 points. Ordinarily this would scream overfitting, but thefocus of our parameters is at different levels, and in the hierarchial set up, 10 of these parameters are really pooled together from one sigma. So the effective number of parameters is something lower.

<!-- cell:72 type:code -->
```python
az.plot_trace(trace3)
```
Output:
```
array([[<Axes: title={'center': 'betap'}>,
        <Axes: title={'center': 'betap'}>],
       [<Axes: title={'center': 'alpha'}>,
        <Axes: title={'center': 'alpha'}>],
       [<Axes: title={'center': 'alphasoc'}>,
        <Axes: title={'center': 'alphasoc'}>],
       [<Axes: title={'center': 'sigmasoc'}>,
        <Axes: title={'center': 'sigmasoc'}>]], dtype=object)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-49-output-2.png)

<!-- cell:73 type:code -->
```python
np.mean(trace3.sample_stats['diverging'].values)
```
Output:
```
np.float64(0.0)
```

<!-- cell:74 type:code -->
```python
az.summary(trace3)
```
Output:
```
              mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
betap        0.260  0.082   0.104    0.416      0.001    0.001    8767.0    7947.0    1.0
alpha        3.442  0.126   3.207    3.679      0.002    0.002    7387.0    7646.0    1.0
alphasoc[0] -0.204  0.245  -0.670    0.244      0.002    0.003   11983.0   10493.0    1.0
alphasoc[1]  0.046  0.221  -0.355    0.486      0.002    0.002   10082.0   10852.0    1.0
alphasoc[2] -0.046  0.197  -0.419    0.327      0.002    0.002   12636.0   10971.0    1.0
alphasoc[3]  0.331  0.193  -0.026    0.692      0.002    0.002    8312.0   10502.0    1.0
alphasoc[4]  0.047  0.180  -0.293    0.387      0.002    0.002   11440.0   11203.0    1.0
alphasoc[5] -0.321  0.211  -0.703    0.074      0.002    0.002   10461.0   10778.0    1.0
alphasoc[6]  0.147  0.177  -0.187    0.479      0.002    0.002   10068.0   11227.0    1.0
alphasoc[7] -0.170  0.186  -0.526    0.181      0.002    0.002   11203.0    8724.0    1.0
alphasoc[8]  0.278  0.179  -0.056    0.614      0.002    0.002    8798.0    9830.0    1.0
alphasoc[9] -0.092  0.297  -0.656    0.483      0.003    0.004    8845.0    7786.0    1.0
sigmasoc     0.313  0.130   0.097    0.551      0.002    0.002    4570.0    5225.0    1.0
```

<!-- cell:75 type:markdown -->
We can ask the WAIC how many effective parameters it has, and it tells us roughly 5. Thus you really care about the number of hyper-parameters you have, and not so much about the lower level parameters.

<!-- cell:76 type:code -->
```python
az.waic(trace3)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
Computed from 20000 posterior samples and 10 observations log-likelihood matrix.

          Estimate       SE
elpd_waic   -34.94     1.28
p_waic        4.94        -

There has been a warning during the calculation. Please check the results.
```

<!-- cell:77 type:markdown -->
We now write code where now we use sampling from the normal corresponding to $\sigma_{society}$ to simulate our societies. Again, we dont use theano's shareds, opting simply to generate samples for the residual intercepts for multiple societies. How many? As many as the traces. You might have thought you only need to generate as many as there are grid points, ie 30, but at the end the posterior predictive must marginalize over the traces at all these points, and thus marginalizing over the full trace at each point suffices!

<!-- cell:78 type:code -->
```python
def compute_pp2(lpgrid, idata, contact=0):
    alphatrace = trace_or_zero(idata, 'alpha')
    betaptrace = trace_or_zero(idata, 'betap')
    sigmasoctrace = trace_or_zero(idata, 'sigmasoc')
    tl = len(alphatrace)
    gl = lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid_c = lpgrid - lpgrid.mean()
    #simulate. alphasocs generated here
    alphasoctrace = np.random.normal(0, sigmasoctrace)
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid_c[i] + alphasoctrace
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```

<!-- cell:79 type:code -->
```python
ppsamps = compute_pp2(lpgrid, trace3)

```

<!-- cell:80 type:code -->
```python
ppmed = np.median(ppsamps, axis=1)
pphpd = az.hdi(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
```
Output:
```
/var/folders/wq/mr3zj9r14dzgjnq9rjx_vqbc0000gn/T/ipykernel_99269/3714602409.py:2: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
  pphpd = az.hdi(ppsamps.T)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-55-output-2.png)

<!-- cell:81 type:markdown -->
The envelope of predictions is much wider here, but overlaps all the points! This is because of the varying intercepts, and it reflects the fact that there is much more variation in the data than is expected from a pure poisson model.

<!-- cell:82 type:markdown -->
## Cross Validation and stacking BMA in pymc

<!-- cell:83 type:code -->
```python
comparedf = az.compare(compare_dict, ic="loo", method="pseudo-BMA")
comparedf.head()
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
  warnings.warn(
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
  warnings.warn(
```
Output:
```
            rank   elpd_loo      p_loo  elpd_diff        weight         se        dse  warning scale
m2c_nopc       0 -39.885648   4.588078   0.000000  9.398024e-01   5.537213   0.000000     True   log
m2c_onlyp      1 -42.853990   4.381984   2.968343  4.829495e-02   4.495307   3.978302     True   log
m1c            2 -44.254554   9.257126   4.368906  1.190268e-02   6.535852   2.657803     True   log
m2c_onlyic     3 -70.966581   8.499973  31.080933  2.983725e-14  15.939870  16.284487     True   log
m2c_onlyc      4 -75.538210  17.092756  35.652563  3.085498e-16  22.478023  22.088001     True   log
```

<!-- cell:84 type:code -->
```python
# az.compare already uses model names as index
comparedf
```
Output:
```
            rank   elpd_loo      p_loo  elpd_diff        weight         se        dse  warning scale
m2c_nopc       0 -39.885648   4.588078   0.000000  9.398024e-01   5.537213   0.000000     True   log
m2c_onlyp      1 -42.853990   4.381984   2.968343  4.829495e-02   4.495307   3.978302     True   log
m1c            2 -44.254554   9.257126   4.368906  1.190268e-02   6.535852   2.657803     True   log
m2c_onlyic     3 -70.966581   8.499973  31.080933  2.983725e-14  15.939870  16.284487     True   log
m2c_onlyc      4 -75.538210  17.092756  35.652563  3.085498e-16  22.478023  22.088001     True   log
```

<!-- cell:85 type:code -->
```python
az.plot_compare(comparedf)
```
![Figure](https://rahuldave.com/posts/islands2/index_files/figure-html/cell-58-output-1.png)

<!-- cell:86 type:code -->
```python
comparedf_s = az.compare(compare_dict, ic="waic", method="stacking")
comparedf_s.head()
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
/Users/rahul/Library/Caches/uv/archive-v0/aTiHGxSE8gD8G3bEQyxJO/lib/python3.14/site-packages/arviz/stats/stats.py:1652: UserWarning: For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. 
See http://arxiv.org/abs/1507.04544 for details
  warnings.warn(
```
Output:
```
            rank  elpd_waic     p_waic  elpd_diff    weight         se        dse  warning scale
m2c_nopc       0 -39.493347   4.195777   0.000000  0.760466   5.528580   0.000000     True   log
m1c            1 -41.976041   6.978613   2.482694  0.000000   6.064680   1.834442     True   log
m2c_onlyp      2 -42.202093   3.730087   2.708746  0.239534   4.469968   3.961737     True   log
m2c_onlyic     3 -70.740050   8.273443  31.246703  0.000000  15.812725  16.368695     True   log
m2c_onlyc      4 -75.175479  16.730025  35.682132  0.000000  22.393098  22.251583     True   log
```

<!-- cell:87 type:code -->
```python
# az.compare already uses model names as index
comparedf_s
```
Output:
```
            rank  elpd_waic     p_waic  elpd_diff    weight         se        dse  warning scale
m2c_nopc       0 -39.493347   4.195777   0.000000  0.760466   5.528580   0.000000     True   log
m1c            1 -41.976041   6.978613   2.482694  0.000000   6.064680   1.834442     True   log
m2c_onlyp      2 -42.202093   3.730087   2.708746  0.239534   4.469968   3.961737     True   log
m2c_onlyic     3 -70.740050   8.273443  31.246703  0.000000  15.812725  16.368695     True   log
m2c_onlyc      4 -75.175479  16.730025  35.682132  0.000000  22.393098  22.251583     True   log
```
