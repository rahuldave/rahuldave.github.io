<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "seaborn",
# ]
# ///

```

<!-- cell:2 type:code -->
```python
%matplotlib inline
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
```

<!-- cell:3 type:markdown -->
## Formulation of the problem

This problem, taken from McElreath's book, involves a seal (or a well trained human) tossing a globe, catching it  on the nose, and noting down if the globe came down on  water or land.

The seal tells us that the first 9 samples were:

`WLWWWLWLW`.

We wish to understand the evolution of belief in the fraction of water on earth as the seal tosses the globe.

Suppose $\theta$ is the true fraction of  water covering the globe. Our data story if that $\theta$ then is the probability of the nose landing on water, with each throw or toss of the globe being independent.

Now we build a  probabilistic model for the problem, which we shall use to guide a process of **Bayesian updating** of the model as data comes in.

$$\cal{L} = p(n,k|\theta) = Binom(n,k, \theta)=\frac{n!}{k! (n-k)! } \, \theta^k \, (1-\theta)^{(n-k)} $$

Since our seal hasnt really seen any water or land, (strange, I know), it assigns equal probabilities, ie uniform probability to any value of $\theta$.

**This is our prior information**

For reasons of conjugacy we 
choose as prior the beta distribution, with $Beta(1,1)$ being the uniform prior.

<!-- cell:4 type:markdown -->
## How to do the Bayesian Process

Bayes theorem and the things we will go through

(1) Grid approximation
(2) Quadratic (Laplace) Approximation
(3) Conjugate Priors
(4) MCMC (later)
(5) Model Checking

<!-- cell:5 type:markdown -->
## Grid Approximation

<!-- cell:6 type:code -->
```python
from scipy.stats import binom
```

<!-- cell:7 type:code -->
```python
prior_pdf = lambda p: 1
like_pdf = lambda p: binom.pmf(k=6, n=9, p=p)
post_pdf = lambda p: like_pdf(p)*prior_pdf(p)
```

<!-- cell:8 type:code -->
```python
p_grid = np.linspace(0., 1., 20)
p_grid
```
Output:
```
array([0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
       0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
       0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
       0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ])
```

<!-- cell:9 type:code -->
```python
plt.plot(p_grid, post_pdf(p_grid),'o-');
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-7-output-1.png)

<!-- cell:10 type:code -->
```python
p_grid = np.linspace(0., 1., 1000)
post_vals = post_pdf(p_grid)
post_vals_normed = post_vals/np.sum(post_vals)
grid_post_samples = np.random.choice(p_grid, size=10000, replace=True, p=post_vals_normed)
```

<!-- cell:11 type:code -->
```python
plt.plot(p_grid, post_vals)
```
Output:
```
[<matplotlib.lines.Line2D at 0x120f08590>]
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-9-output-1.png)

<!-- cell:12 type:code -->
```python
sns.histplot(grid_post_samples, kde=True)
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-10-output-1.png)

<!-- cell:13 type:markdown -->
## Laplace Approximation

<!-- cell:14 type:code -->
```python
p_start = 0.5
from scipy.optimize import minimize
post_pdf_inv = lambda p: -post_pdf(p)
res = minimize(post_pdf_inv, p_start, method = 'Nelder-Mead', options={'disp': True})
```
Output:
```
Optimization terminated successfully.
         Current function value: -0.273129
         Iterations: 13
         Function evaluations: 26
```

<!-- cell:15 type:code -->
```python
res
```
Output:
```
       message: Optimization terminated successfully.
       success: True
        status: 0
           fun: -0.2731290903134584
             x: [ 6.667e-01]
           nit: 13
          nfev: 26
 final_simplex: (array([[ 6.667e-01],
                       [ 6.666e-01]]), array([-2.731e-01, -2.731e-01]))
```

<!-- cell:16 type:code -->
```python
post_MAP = res.x[0]
post_MAP
```
Output:
```
np.float64(0.6666992187500004)
```

<!-- cell:17 type:code -->
```python
insertbefore = np.searchsorted(p_grid, post_MAP)
insertbefore
```
Output:
```
np.int64(667)
```

<!-- cell:18 type:code -->
```python
postmapval = (post_vals[insertbefore-1] + post_vals[insertbefore])/2.
postmapval
```
Output:
```
np.float64(0.2731263224481272)
```

<!-- cell:19 type:code -->
```python
plt.plot(p_grid, post_vals);
plt.plot(p_grid, norm.pdf(p_grid, loc=post_MAP, scale=0.16))
```
Output:
```
[<matplotlib.lines.Line2D at 0x121029940>]
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-16-output-1.png)

<!-- cell:20 type:code -->
```python
zq = lambda sigma: sigma*postmapval*np.sqrt(2*np.pi)
def fit_loss(sigma):
    vec = (post_vals/zq(sigma)) - norm.pdf(p_grid, loc=post_MAP, scale=sigma)
    return np.dot(vec, vec)
```

<!-- cell:21 type:code -->
```python
res2 = minimize(fit_loss, 0.2, method = 'Nelder-Mead', options={'disp': True})
```
Output:
```
Optimization terminated successfully.
         Current function value: 23.987144
         Iterations: 12
         Function evaluations: 24
```

<!-- cell:22 type:code -->
```python
res2
```
Output:
```
       message: Optimization terminated successfully.
       success: True
        status: 0
           fun: 23.98714369935766
             x: [ 1.492e-01]
           nit: 12
          nfev: 24
 final_simplex: (array([[ 1.492e-01],
                       [ 1.492e-01]]), array([ 2.399e+01,  2.399e+01]))
```

<!-- cell:23 type:code -->
```python
post_SIG = res2.x[0]
post_SIG
```
Output:
```
np.float64(0.1492187500000001)
```

<!-- cell:24 type:code -->
```python
frozen_laplace = norm(post_MAP, post_SIG)
```

<!-- cell:25 type:code -->
```python
plt.plot(p_grid, post_pdf(p_grid)/zq(post_SIG), label = "normalized posterior");
plt.plot(p_grid, frozen_laplace.pdf(p_grid), label = "laplace approx")
plt.legend();
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-22-output-1.png)

<!-- cell:26 type:code -->
```python
zq(post_SIG)
```
Output:
```
np.float64(0.10215906016979828)
```

<!-- cell:27 type:markdown -->
Now we can get samples from here:

<!-- cell:28 type:code -->
```python
sns.histplot(frozen_laplace.rvs(10000), kde=True)
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-24-output-1.png)

<!-- cell:29 type:markdown -->
## Conjugate Priors

The  mean of $Beta(\alpha, \beta)$ is  $\mu = \frac{\alpha}{\alpha+\beta}$ while the variance is 

$$V=\mu (1- \mu)/(\alpha + \beta + 1)$$

<!-- cell:30 type:code -->
```python
from scipy.stats import beta
x=np.linspace(0., 1., 100)
plt.plot(x, beta.pdf(x, 1, 1));
plt.plot(x, beta.pdf(x, 1, 9));
plt.plot(x, beta.pdf(x, 1.2, 9));
plt.plot(x, beta.pdf(x, 2, 18));
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-25-output-1.png)

<!-- cell:31 type:markdown -->
We shall choose $\alpha=1$ and $\beta=1$ to be uniform.

<!-- cell:32 type:markdown -->
$$ p(\theta) = {\rm Beta}(\theta,\alpha, \beta) = \frac{\theta^{\alpha-1} (1-x)^{\beta-1} }{B(\alpha, \beta)} $$
where $B(\alpha, \beta)$ is independent of $\theta$ and it is the normalization factor.

From Bayes theorem, the posterior for $\theta$ is 

$$ p(\theta|D) \propto  p(\theta) \, p(n,k|\theta)  =  Binom(n,k, \theta) \,  {\rm Beta}(\theta,\alpha, \beta)  $$

which can be shown to be 

$${\rm Beta}(\theta, \alpha+k, \beta+n-k)$$

<!-- cell:33 type:code -->
```python
from scipy.stats import beta, binom

plt.figure(figsize=( 15, 18))

prior_params = np.array( [1.,1.] )  # FLAT 

x = np.linspace(0.00, 1, 125)
datastring = "WLWWWLWLW"
data=[]
for c in datastring:
    data.append(1*(c=='W'))
data=np.array(data)
print(data)
choices=['Land','Water']


for i,v in enumerate(data):
    plt.subplot(9,1,i+1)
    prior_pdf = beta.pdf( x, *prior_params)
    if v==1:
        water = [1,0]
    else:
        water = [0,1]
    posterior_params = prior_params + np.array( water )    # posteriors beta parameters
    posterior_pdf = beta.pdf( x, *posterior_params)  # the posterior 
    prior_params = posterior_params
    plt.plot( x,prior_pdf, label = r"prior for this step", lw =1, color ="#348ABD" )
    plt.plot( x, posterior_pdf, label = "posterior for this step", lw= 3, color ="#A60628" )
    plt.fill_between( x, 0, prior_pdf, color ="#348ABD", alpha = 0.15) 
    plt.fill_between( x, 0, posterior_pdf, color ="#A60628", alpha = 0.15) 
    
    plt.legend(title = "N=%d, %s"%(i, choices[v]));
    #plt.ylim( 0, 10)#
```
Output:
```
[1 0 1 1 1 0 1 0 1]
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-26-output-2.png)

<!-- cell:34 type:markdown -->
## Interrogating the posterior

Since we can sample from the posterior now after 9 observations, lets do so!

<!-- cell:35 type:code -->
```python
samples = beta.rvs(*posterior_params, size=10000)
plt.hist(samples, bins=50, density=True);
sns.kdeplot(samples);
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-27-output-1.png)

<!-- cell:36 type:markdown -->
### Sampling to summarize

Now we can calculate all sorts of stuff.

The probability that the amount of water is less than 50%

<!-- cell:37 type:code -->
```python
np.mean(samples < 0.5)
```
Output:
```
np.float64(0.174)
```

<!-- cell:38 type:markdown -->
The probability by which we get 80% of the samples.

<!-- cell:39 type:code -->
```python
np.percentile(samples, 80)
```
Output:
```
np.float64(0.7611465557699019)
```

<!-- cell:40 type:markdown -->
You might try and find a **credible interval**. This, unlike the wierd definition of confidence intervals, is exactly what you think it is, the amount of probability mass between certain percentages, like the middle 95%

<!-- cell:41 type:code -->
```python
np.percentile(samples, [2.5, 97.5])
```
Output:
```
array([0.34690217, 0.87950578])
```

<!-- cell:42 type:markdown -->
You can make various point estimates: mean, median

<!-- cell:43 type:code -->
```python
np.mean(samples), np.median(samples), np.percentile(samples, 50) #last 2 are same
```
Output:
```
(np.float64(0.6356707835604966),
 np.float64(0.6424956252802423),
 np.float64(0.6424956252802423))
```

<!-- cell:44 type:markdown -->
A particularly important and useful point estimate that we just saw is the **MAP**, or "maximum a-posteriori" estimate, the value of the parameter at which the pdf (num-samples) reach a maximum. It can be obtained from the samples as well.

<!-- cell:45 type:code -->
```python
sampleshisto = np.histogram(samples, bins=50)
```

<!-- cell:46 type:code -->
```python
maxcountindex = np.argmax(sampleshisto[0])
mapvalue = sampleshisto[1][maxcountindex]
print(maxcountindex, mapvalue)
```
Output:
```
32 0.680260779862307
```

<!-- cell:47 type:markdown -->
The mean of the posterior samples corresponds to minimizing the squared loss.

<!-- cell:48 type:code -->
```python
mse = [np.mean((xi-samples)**2) for xi in x]
plt.plot(x, mse);
plt.axvline(np.mean(samples), 0, 1, color="r")
print("Mean",np.mean(samples));
```
Output:
```
Mean 0.6356707835604966
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-34-output-2.png)

<!-- cell:49 type:markdown -->
## Sampling to simulate prediction: the posterior predictive

Why would you want to simulate prediction?

1. Model Checking
2. Software Validation
3. Research Design
4. Forecasting

Its easy to sample from any one probability to get the sampling distribution at a particular $\theta$

<!-- cell:50 type:code -->
```python
point3samps = np.random.binomial( len(data), 0.3, size=10000);
point7samps = np.random.binomial( len(data), 0.7, size=10000);
plt.hist(point3samps, lw=3, alpha=0.5, histtype="stepfilled", bins=np.arange(11));
plt.hist(point7samps, lw=3, alpha=0.3,histtype="stepfilled", bins=np.arange(11));
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-35-output-1.png)

<!-- cell:51 type:markdown -->
The posterior predictive:

$$p(y^{*} \vert D) = \int d\theta p(y^{*} \vert \theta) p(\theta \vert D)$$

seems to be a complex integral.  But if you parse it, its not so complex. This diagram from McElreath helps:

![The posterior predictive distribution as a mixture: each parameter value implies a sampling distribution, weighted by the posterior probability, producing the marginal prediction. From McElreath, Statistical Rethinking.](https://rahuldave.com/posts/globemodellab/assets/postpred.png)


### Plug-in Approximation

Also, often, people will use the **plug-in approximation** by putting the posterior mean or MAP value 

$$p(\theta \vert D) = \delta(\theta - \theta_{MAP})$$

and then simply  drawing the posterior predictive  from :

$$p(y^{*} \vert D) = p(y^{*} \vert \theta_{MAP})$$

(the same thing could be done for $\theta_{mean}$).

<!-- cell:52 type:code -->
```python
pluginpreds = np.random.binomial( len(data), mapvalue, size = len(samples))
```

<!-- cell:53 type:code -->
```python
plt.hist(pluginpreds, bins=np.arange(11));
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-37-output-1.png)

<!-- cell:54 type:markdown -->
This approximation is just sampling from the likelihood(sampling distribution), at a posterior-obtained value of $\theta$.  It might be useful if the posterior is an expensive MCMC and the MAP is easier to find by optimization, and can be used in conjunction with quadratic (gaussian) approximations to the posterior, as we will see in variational inference. But for now we have all the samples, and it would be inane not to use them...

<!-- cell:55 type:markdown -->
### The posterior predictive from sampling

But really from the perspective of sampling, all we have to do is to first draw the thetas from the posterior, then draw y's from the likelihood, and histogram the likelihood. This is the same logic as marginal posteriors, with the addition of the fact that we must draw  y from the likelihood once we drew $\theta$. You might think that we have to draw multiple $y$s at a theta, but this is already taken care of for us because of the nature of sampling. We already have multiple $\theta$s in a bin.

<!-- cell:56 type:code -->
```python
postpred = np.random.binomial( len(data), samples);
```

<!-- cell:57 type:code -->
```python
postpred
```
Output:
```
array([7, 9, 5, ..., 4, 5, 1], shape=(10000,))
```

<!-- cell:58 type:code -->
```python
samples.shape, postpred.shape
```
Output:
```
((10000,), (10000,))
```

<!-- cell:59 type:code -->
```python
plt.hist(postpred, bins=np.arange(11), alpha=0.5, align="left", label="predictive")
plt.hist(pluginpreds, bins=np.arange(11), alpha=0.2, align="left", label="plug-in (MAP)")
plt.title('Posterior predictive')
plt.xlabel('k')
plt.legend()
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-41-output-1.png)

<!-- cell:60 type:markdown -->
### Replicative predictives

There is a different kind of predictive sampling that us useful (and what you might have thought was predictive sampling). This is replicative sampling. It can be used with both priors and posteriors; the former for model callibration and the latter for model checking. We shall see both of these soon.

The idea here is to generate an entire dataset from one of the parameter samples in the posterior. So you are not generating 10000 ys for 10000 thetas, but rather 10000 y's per theta. (you can play the same game with the prior). This kind of inverts the diagram we saw earlier to produce the posterior predictive.

Our usual sample vs replication 2D setup can come useful here. Consider generating 1000 y's per replication for each theta.

<!-- cell:61 type:code -->
```python
postpred.shape
```
Output:
```
(10000,)
```

<!-- cell:62 type:code -->
```python
reppostpred =np.empty((1000, 10000))
for i in range(1000):
    reppostpred[i,:] = np.random.binomial( len(data), samples);
reppostpred.shape
```
Output:
```
(1000, 10000)
```

<!-- cell:63 type:code -->
```python
per_theta_avgs = np.mean(reppostpred, axis=0)
per_theta_avgs.shape
```
Output:
```
(10000,)
```

<!-- cell:64 type:code -->
```python
plt.scatter(samples, per_theta_avgs, alpha=0.1);
```
![Figure](https://rahuldave.com/posts/globemodellab/index_files/figure-html/cell-45-output-1.png)

<!-- cell:65 type:markdown -->
In particular, you will find that the number of switches is not consistent with what you see in our data. This might lead you to question our model...always a good thing..but note that we have very little data as yet to go on

<!-- cell:66 type:code -->
```python
data
```
Output:
```
array([1, 0, 1, 1, 1, 0, 1, 0, 1])
```

<!-- cell:67 type:code -->
```python
data[:-1] != data[1:]
```
Output:
```
array([ True,  True, False, False,  True,  True,  True,  True])
```

<!-- cell:68 type:code -->
```python
np.sum(data[:-1] != data[1:])
```
Output:
```
np.int64(6)
```

<!-- cell:69 type:markdown -->
## Exercise

You can interrogate the posterior-predictive, or **simulated** samples in other ways, asking about the longest run of water tosses, or the number of times the water/land switched. This is left as an exercise. 
