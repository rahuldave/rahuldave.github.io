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
# Gelman Schools and Hierarchical Pathology

##### Keywords: normal-normal model, hierarchical normal-normal model, hierarchical, divergences, non-centered hierarchical model, sampling distribution

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
From Gelman:

>a simple hierarchical model based on the normal distribu- tion, in which observed data are normally distributed with a different mean for each ‘group’ or ‘experiment,’ with known observation variance, and a normal population distribution for the group means. This model is sometimes termed the one-way normal random-effects model with known data variance and is widely applicable, being an important special case of the hierarchical normal linear model,...

## Statement of the model

The particular example we will deal with is called the 8-schools example, and is described thus:

>A study was performed for the Educational Testing Service to analyze the effects of special coaching programs on test scores. Separate randomized experiments were performed to estimate the effects of coaching programs for the SAT-V (Scholastic Aptitude Test- Verbal) in each of eight high schools. The outcome variable in each study was the score on a special administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more effective than any other or that some were more similar in effect to each other than to any other.

>the estimated coaching effects are $\bar{y}_j$, and their sampling variances, $\sigma_j^2$... The estimates $\bar{y}_j$ are obtained by independent experiments and have approximately normal sampling distributions with sampling variances that are known, for all practical purposes, because the sample sizes in all of the eight experiments were relatively large, over thirty students in each school 

![Estimated treatment effects and standard errors for the 8 schools [Source: Gelman et al., Bayesian Data Analysis]](assets/8schools.png)

<!-- cell:5 type:markdown -->
Notice the bar on the y's and the mention of standard errors (rather than standard deviations) in the third column in the table above. Why is this?

The answer is that these are means taken over many (> 30) students in each of the schools. The general structure of this model can be written thus:

>Consider $J$ independent experiments, with experiment $j$ estimating the parameter $\theta_j$ from $n_j$ independent normally distributed data points, $y_{ij}$, each with known error variance $\sigma^2$; that is,

$$y_{ij} \vert \theta_j \sim N(\theta_j, \sigma^2), \, i = 1,...,n_j; j = 1,...,J.$$

<!-- cell:6 type:markdown -->
We label the sample mean of each group $j$ as

$$\bar{y_j} = \frac{1}{n_j} \sum_{i=1}^{n_j} y_{ij}$$

with sampling variance:

$$\sigma_j^2 = \sigma^2/n_j$$

  
>We can then write the likelihood for each $\theta_j$ using the sufficient statistics, $\bar{y}_j$:

$$\bar{y_j} \vert \theta_j \sim N(\theta_j,\sigma_j^2).$$

<!-- cell:7 type:markdown -->
This is

>a notation that will prove useful later because of the flexibility in allowing a separate variance $\sigma_j^2$ for the mean of each group $j$. ...all expressions will be implicitly conditional on the known values $\sigma_j^2$.... Although rarely strictly true, the assumption of known variances at the sampling level of the model is often an adequate approximation.

>The treatment of the model provided ... is also appropriate for situations in which the variances differ for reasons other than the number of data points in the experiment. In fact, the likelihood  can appear in much more general contexts than that stated here. For example, if the group sizes $n_j$ are large enough, then the means $\bar{y_j}$ are approximately normally distributed, given $\theta_j$, even when the data $y_{ij}$ are not. 

<!-- cell:8 type:markdown -->
## Setting up the hierarchical model

We'll set up the modelled in what is called a "Centered" parametrization which tells us how $\theta_i$ is modelled: it is written to be directly dependent as a normal distribution from the hyper-parameters. 

<!-- cell:9 type:code -->
```python
J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])
```

<!-- cell:10 type:markdown -->
We set up our priors in a Hierarchical model to use this centered parametrization. We can say: the $\theta_j$ is drawn from a Normal hyper-prior distribution with parameters $\mu$ and $\tau$. Once we get a $\theta_j$ then can draw the means from it given the data $\sigma_j$ and one such draw corresponds to our data.

<!-- cell:11 type:markdown -->
$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\theta_{j} \sim \mathcal{N}(\mu, \tau)\\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

where $j \in \{1, \ldots, 8 \}$ and the
$\{ y_{j}, \sigma_{j} \}$ are given as data

<!-- cell:12 type:code -->
```python
with pm.Model() as schools1:

    mu = pm.Normal('mu', 0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
```

<!-- cell:13 type:code -->
```python
with schools1:
    idata1 = pm.sample(5000, cores=2, tune=500, random_seed=42)
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
Sampling 2 chains for 500 tune and 5_000 draw iterations (1_000 + 10_000 draws total) took 3 seconds.
```
Output:
```
There were 285 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```
Output:
```
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
```

<!-- cell:14 type:code -->
```python
az.summary(idata1)
```
Output:
```
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu        4.355  3.214  -1.320   10.594      0.081    0.053    1590.0    2885.0   1.00
theta[0]  6.377  5.679  -3.933   17.638      0.120    0.126    2138.0    3219.0   1.00
theta[1]  4.932  4.760  -3.964   13.970      0.090    0.092    2598.0    3875.0   1.00
theta[2]  3.843  5.300  -6.083   14.016      0.099    0.120    2613.0    3647.0   1.00
theta[3]  4.730  4.864  -4.422   14.310      0.087    0.090    2825.0    4420.0   1.00
theta[4]  3.430  4.754  -5.907   12.115      0.103    0.080    2021.0    4088.0   1.00
theta[5]  3.971  4.948  -6.103   12.771      0.097    0.100    2396.0    3769.0   1.01
theta[6]  6.505  5.233  -2.952   16.651      0.113    0.099    2073.0    3200.0   1.00
theta[7]  4.923  5.478  -5.624   15.313      0.091    0.145    2918.0    4445.0   1.01
tau       4.025  3.133   0.383    9.506      0.124    0.088     271.0     147.0   1.02
```

<!-- cell:15 type:markdown -->
The Gelman-Rubin statistic seems fine, but notice how small the effective-n's are? Something is not quite right. Lets see traceplots.

<!-- cell:16 type:code -->
```python
az.plot_trace(idata1);
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-8-output-1.png)

<!-- cell:17 type:markdown -->
Its hard to pick the thetas out but $\tau$ looks not so white-noisy. Lets zoom in:

<!-- cell:18 type:code -->
```python
# In modern pymc, transformed variables like tau_log__ are not directly in the trace.
# We compute log(tau) manually and plot it.
import xarray as xr
logtau_posterior = np.log(idata1.posterior["tau"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for chain in logtau_posterior.chain.values:
    axes[0].plot(logtau_posterior.sel(chain=chain).values, alpha=0.6)
axes[0].set_title("log(tau) trace")
axes[0].set_xlabel("draw")
axes[1].hist(logtau_posterior.values.flatten(), bins=50, alpha=0.7)
axes[1].set_title("log(tau) distribution")
plt.tight_layout()
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-9-output-1.png)

<!-- cell:19 type:markdown -->
There seems to be some stickiness at lower values in the trace. Zooming in even more helps us see this better:

<!-- cell:20 type:code -->
```python
logtau1 = np.log(idata1.posterior["tau"].values.flatten())
plt.plot(logtau1, alpha=0.6)
plt.axvline(5000, color="r")
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-10-output-1.png)

<!-- cell:21 type:markdown -->
We plot the cumulative mean of $log(\tau)$ as time goes on. This definitely shows some problems. Its biased above the value you would expect from many many samples.

<!-- cell:22 type:code -->
```python
# plot the estimate for the mean of log(τ) cumulating mean
logtau = np.log(idata1.posterior["tau"].values.flatten())
mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color='gray')
plt.plot(mlogtau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel('Iteration')
plt.ylabel('MCMC mean of log(tau)')
plt.title('MCMC estimation of cumsum log(tau)')
```
Output:
```
Text(0.5, 1.0, 'MCMC estimation of cumsum log(tau)')
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-11-output-2.png)

<!-- cell:23 type:markdown -->
## The problem with curvature

In-fact the "sticky's" in the traceplot are trying to drag the $\tau$ trace down to the true value, eventually. If we wait for the heat-death of the universe long this will happen, given MCMC's guarantees. But clearly we want to get there with a finite number of samples.

We can diagnose whats going on by looking for divergent traces in pymc. These are obtained from the sample statistics in the InferenceData object.

<!-- cell:24 type:code -->
```python
divergent = idata1.sample_stats["diverging"].values.flatten()
print('Number of Divergent %d' % divergent.nonzero()[0].size)
n_total = idata1.posterior.sizes["chain"] * idata1.posterior.sizes["draw"]
divperc = divergent.nonzero()[0].size / n_total
print('Percentage of Divergent %.5f' % divperc)
```
Output:
```
Number of Divergent 285
Percentage of Divergent 0.02850
```

<!-- cell:25 type:markdown -->
What does divergent mean? These are situations in which our symplectic integrator has gone Kaput, as illustrated in this diagram below from Betancourt's review:

![Symplectic integrator trajectory diverging from the true orbit due to excessive step size [Source: Betancourt 2017]](assets/sympldiv.png)

When a MCMC sampler (this problem is worse in non HMC models) encounters a region of high curvature it gets stuck, and goes off elsewhere, after a while. In the HMC scenario, the symplectic integrator diverges. What happens is that in encounters a region of high curvature which our timestep size $\epsilon$ is not able to resolve. Things become numerically unstable and the integrator veers off towards infinite energies, clearly not conserving energy any more.

![Leapfrog steps (green) diverge from the true trajectory (red) in a region of high curvature [Source: Betancourt 2017]](assets/sympldiv2.png)

<!-- cell:26 type:markdown -->
Where is this curvature coming from? Things become a bit easier to understand if we plot the joint distribution of one of the thetas against a hyper-parameter. And we see that there is a triangular funnel structure with the hyperparameter spanning orders of magnitude in value. Using pymc we can plot where the divergences occur, and while they can occur anywhere they seem to be clustered in the neck of the distribution.

<!-- cell:27 type:code -->
```python
theta_trace = idata1.posterior["theta"].values  # (chains, draws, J)
theta0 = theta_trace[:, :, 0].flatten()
logtau = np.log(idata1.posterior["tau"].values.flatten())
plt.figure(figsize=(10, 6))
plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r', s=10, alpha=0.05)
plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g', s=10, alpha=0.9)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.show()
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-13-output-1.png)

<!-- cell:28 type:markdown -->
Two things have now happened...we have an increasing inability to integrate in the neck of this funnel, and we have lost confidence that our sampler is now actually characterizing this funnel well.

`pymc` warning system also captures this, and the information can be drawn from there as well

<!-- cell:29 type:code -->
```python
# Check if sampling had issues (divergences)
n_divergent = idata1.sample_stats["diverging"].values.sum()
print(f"Sampling OK: {n_divergent == 0}")
print(f"Number of divergences: {n_divergent}")
```
Output:
```
Sampling OK: False
Number of divergences: 285
```

<!-- cell:30 type:code -->
```python
# Show sample stats summary for diagnostics
print("Divergences per chain:")
for chain in idata1.sample_stats.chain.values:
    n_div = idata1.sample_stats["diverging"].sel(chain=chain).values.sum()
    print(f"  Chain {chain}: {n_div} divergences")
```
Output:
```
Divergences per chain:
  Chain 0: 170 divergences
  Chain 1: 115 divergences
```

<!-- cell:31 type:markdown -->
## Funnels in hierarchical models

As is discussed in Betancourt and Girolami (2015) from where the following diagrams are taken, a funnel structure is common in Hierarchical models, and reflects strong correlations between down-tree parameters such as thetas, and uptree parameters such as $\phi = \mu, \tau$ here.

![Graphical model for a hierarchical Bayesian model: data nodes D, group parameters θ, and shared hyperparameter φ [Source: Betancourt & Girolami 2013]](assets/girolam1.png)

The funnel between $v = log(\tau)$ and $\theta_i$ in the hierarchical Normal-Normal model looks like this:

![Neal's funnel: the (100+1)-dimensional posterior concentrates into a narrow neck at small τ, making sampling difficult [Source: Betancourt & Girolami 2013]](assets/girolam2.png)

The problem is that a sampler must sample both the light and dark regions as both have enough probability mass.

### Divergences are good things

This is because they can help us diagnose problems in our samplers. Chances are that in a region with divergences the sampler is having problems exploring.

There is a second reason besides curvature and symplectic integration which affects the efficiency of HMC. This is the range of transitions a HMC sampler can make. For a euclidean mass-matrix sampler, the transitions are of the order of the range in kinetic energy, which itself is chi-squared distributed (since p is Gaussian and a sum of gaussians is a chi-squared). Thus, in expectation, the variation in $K$ is of oder $d/2$ where $d$ is the dimension of the target. Since hierarchical structures correlate variables between levels, they also induce large changes in energy density, which our transitions dont explore well.

## Non-centered Model

So what is to be done? We could change the kinetic energy using methods such as Riemannian Monte-Carlo HMC, but thats beyond our scope. But, just as in the case with the regression example earlier, a re-parametrization comes to our rescue. We want to reduce the levels in the hierarchy, as shown here:

![Centered (left) vs non-centered (right) parametrization: decoupling θ and φ removes the funnel geometry [Source: Betancourt & Girolami 2013]](assets/girolam3.png)

<!-- cell:32 type:markdown -->
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

<!-- cell:33 type:code -->
```python
with pm.Model() as schools2:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Normal('nu', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * nu)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
```

<!-- cell:34 type:code -->
```python
with schools2:
    idata2 = pm.sample(5000, cores=2, tune=500, random_seed=42)
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
Sampling 2 chains for 500 tune and 5_000 draw iterations (1_000 + 10_000 draws total) took 1 seconds.
```
Output:
```
There were 8 divergences after tuning. Increase `target_accept` or reparameterize.
```
Output:
```
We recommend running at least 4 chains for robust computation of convergence diagnostics
```

<!-- cell:35 type:code -->
```python
az.summary(idata2)
```
Output:
```
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu        4.374  3.298  -1.730   10.650      0.032    0.035   10449.0    7025.0    1.0
nu[0]     0.315  1.007  -1.617    2.153      0.010    0.010    9937.0    6969.0    1.0
nu[1]     0.090  0.941  -1.738    1.790      0.009    0.010   11177.0    7312.0    1.0
nu[2]    -0.086  0.972  -1.877    1.779      0.009    0.010   11112.0    6973.0    1.0
nu[3]     0.056  0.937  -1.734    1.797      0.008    0.011   14082.0    6552.0    1.0
nu[4]    -0.150  0.945  -1.926    1.675      0.009    0.010   10270.0    6829.0    1.0
nu[5]    -0.079  0.936  -1.778    1.724      0.009    0.009   10732.0    7139.0    1.0
nu[6]     0.369  0.976  -1.570    2.114      0.009    0.010   11962.0    7443.0    1.0
nu[7]     0.074  0.965  -1.733    1.858      0.009    0.010   12298.0    7422.0    1.0
tau       3.593  3.189   0.001    9.478      0.038    0.051    6064.0    4854.0    1.0
theta[0]  6.205  5.629  -3.974   16.647      0.060    0.084    9652.0    7279.0    1.0
theta[1]  4.901  4.647  -3.978   13.615      0.043    0.056   12168.0    7516.0    1.0
theta[2]  3.874  5.373  -6.081   14.077      0.056    0.078   10164.0    6546.0    1.0
theta[3]  4.774  4.778  -4.109   14.056      0.046    0.057   11408.0    7676.0    1.0
theta[4]  3.592  4.643  -5.240   12.459      0.045    0.052   11079.0    7676.0    1.0
theta[5]  3.976  4.810  -5.029   13.434      0.048    0.056   10543.0    7551.0    1.0
theta[6]  6.312  5.013  -2.772   15.986      0.049    0.061   11141.0    8013.0    1.0
theta[7]  4.744  5.254  -5.362   14.738      0.053    0.075   10535.0    7226.0    1.0
```

<!-- cell:36 type:code -->
```python
az.plot_trace(idata2);
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-19-output-1.png)

<!-- cell:37 type:code -->
```python
logtau2_posterior = np.log(idata2.posterior["tau"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for chain in logtau2_posterior.chain.values:
    axes[0].plot(logtau2_posterior.sel(chain=chain).values, alpha=0.6)
axes[0].set_title("log(tau) trace")
axes[0].set_xlabel("draw")
axes[1].hist(logtau2_posterior.values.flatten(), bins=50, alpha=0.7)
axes[1].set_title("log(tau) distribution")
plt.tight_layout()
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-20-output-1.png)

<!-- cell:38 type:markdown -->
Ok, so this seems to look better!

<!-- cell:39 type:code -->
```python
logtau2 = np.log(idata2.posterior["tau"].values.flatten())
plt.plot(logtau2, alpha=0.6)
plt.axvline(5000, color="r")
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-21-output-1.png)

<!-- cell:40 type:markdown -->
And the effective number of iterations hs improved as well:

<!-- cell:41 type:code -->
```python
az.rhat(idata2), az.ess(idata2)
```
Output:
```
(<xarray.Dataset> Size: 272B
 Dimensions:      (nu_dim_0: 8, theta_dim_0: 8)
 Coordinates:
   * nu_dim_0     (nu_dim_0) int64 64B 0 1 2 3 4 5 6 7
   * theta_dim_0  (theta_dim_0) int64 64B 0 1 2 3 4 5 6 7
 Data variables:
     mu           float64 8B 1.0
     nu           (nu_dim_0) float64 64B 1.0 1.0 0.9999 1.001 1.0 1.001 1.001 1.0
     tau          float64 8B 1.0
     theta        (theta_dim_0) float64 64B 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
 Attributes:
     created_at:                 2026-03-06T17:10:20.539910+00:00
     arviz_version:              0.23.4
     inference_library:          pymc
     inference_library_version:  5.28.1
     sampling_time:              1.1466081142425537
     tuning_steps:               500,
 <xarray.Dataset> Size: 272B
 Dimensions:      (nu_dim_0: 8, theta_dim_0: 8)
 Coordinates:
   * nu_dim_0     (nu_dim_0) int64 64B 0 1 2 3 4 5 6 7
   * theta_dim_0  (theta_dim_0) int64 64B 0 1 2 3 4 5 6 7
 Data variables:
     mu           float64 8B 1.045e+04
     nu           (nu_dim_0) float64 64B 9.937e+03 1.118e+04 ... 1.23e+04
     tau          float64 8B 6.064e+03
     theta        (theta_dim_0) float64 64B 9.652e+03 1.217e+04 ... 1.054e+04
 Attributes:
     created_at:                 2026-03-06T17:10:20.539910+00:00
     arviz_version:              0.23.4
     inference_library:          pymc
     inference_library_version:  5.28.1
     sampling_time:              1.1466081142425537
     tuning_steps:               500)
```

<!-- cell:42 type:markdown -->
And we reach the true value better as the number of samples increases, decreasing our bias

<!-- cell:43 type:code -->
```python
# plot the estimate for the mean of log(τ) cumulating mean
logtau = np.log(idata2.posterior["tau"].values.flatten())
mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color='gray')
plt.plot(mlogtau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel('Iteration')
plt.ylabel('MCMC mean of log(tau)')
plt.title('MCMC estimation of cumsum log(tau)')
```
Output:
```
Text(0.5, 1.0, 'MCMC estimation of cumsum log(tau)')
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-23-output-2.png)

<!-- cell:44 type:markdown -->
How about our divergences? They have decreased too.

<!-- cell:45 type:code -->
```python
divergent = idata2.sample_stats["diverging"].values.flatten()
print('Number of Divergent %d' % divergent.nonzero()[0].size)
n_total = idata2.posterior.sizes["chain"] * idata2.posterior.sizes["draw"]
divperc = divergent.nonzero()[0].size / n_total
print('Percentage of Divergent %.5f' % divperc)
```
Output:
```
Number of Divergent 8
Percentage of Divergent 0.00080
```

<!-- cell:46 type:code -->
```python
theta_trace = idata2.posterior["theta"].values  # (chains, draws, J)
theta0 = theta_trace[:, :, 0].flatten()
logtau = np.log(idata2.posterior["tau"].values.flatten())
plt.figure(figsize=(10, 6))
plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r', s=10, alpha=0.05)
plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g', s=20, alpha=0.9)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.show()
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-25-output-1.png)

<!-- cell:47 type:markdown -->
Look how much longer the funnel actually is. And we have explored this much better.

<!-- cell:48 type:code -->
```python
theta01 = idata1.posterior["theta"].values[:, :, 0].flatten()
logtau1 = np.log(idata1.posterior["tau"].values.flatten())

theta02 = idata2.posterior["theta"].values[:, :, 0].flatten()
logtau2 = np.log(idata2.posterior["tau"].values.flatten())


plt.figure(figsize=(10, 6))
plt.scatter(theta01, logtau1, alpha=.05, color="b", label="original", s=10)
plt.scatter(theta02, logtau2, alpha=.05, color='r', label='reparametrized', s=10)
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.legend()
```
![Figure](https://rahuldave.com/posts/gelmanschools/index_files/figure-html/cell-26-output-1.png)

<!-- cell:49 type:markdown -->
It may not be possible in all models to achieve this sort of decoupling. In that case, Riemannian HMC, where we generalize the mass matrix to depend upon position, explicitly tackling high-curvature, can help.
