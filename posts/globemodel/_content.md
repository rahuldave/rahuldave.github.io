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
## Choosing a prior and posterior

The  mean of $Beta(\alpha, \beta)$ is  $\mu = \frac{\alpha}{\alpha+\beta}$ while the variance is 

$$V=\mu (1- \mu)/(\alpha + \beta + 1)$$

<!-- cell:5 type:code -->
```python
from scipy.stats import beta
x=np.linspace(0., 1., 100)
plt.plot(x, beta.pdf(x, 1, 1));
plt.plot(x, beta.pdf(x, 1, 9));
plt.plot(x, beta.pdf(x, 1.2, 9));
plt.plot(x, beta.pdf(x, 2, 18));
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-4-output-1.png)

<!-- cell:6 type:markdown -->
We shall choose $\alpha=1$ and $\beta=1$ to be uniform.

<!-- cell:7 type:markdown -->
$$ p(\theta) = {\rm Beta}(\theta,\alpha, \beta) = \frac{\theta^{\alpha-1} (1-x)^{\beta-1} }{B(\alpha, \beta)} $$
where $B(\alpha, \beta)$ is independent of $\theta$ and it is the normalization factor.

From Bayes theorem, the posterior for $\theta$ is 

$$ p(\theta|D) \propto  p(\theta) \, p(n,k|\theta)  =  Binom(n,k, \theta) \,  {\rm Beta}(\theta,\alpha, \beta)  $$

which can be shown to be 

$${\rm Beta}(\theta, \alpha+k, \beta+n-k)$$

<!-- cell:8 type:code -->
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
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-5-output-2.png)

<!-- cell:9 type:markdown -->
## Interrogating the posterior

Since we can sample from the posterior now after 9 observations, lets do so!

<!-- cell:10 type:code -->
```python
samples = beta.rvs(*posterior_params, size=10000)
plt.hist(samples, bins=50, density=True);
sns.kdeplot(samples);
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-6-output-1.png)

<!-- cell:11 type:markdown -->
Now we can calculate all sorts of stuff.

The probability that the amount of water is less than 50%

<!-- cell:12 type:code -->
```python
np.mean(samples < 0.5)
```
Output:
```
np.float64(0.1705)
```

<!-- cell:13 type:markdown -->
The probability by which we get 80% of the samples.

<!-- cell:14 type:code -->
```python
np.percentile(samples, 80)
```
Output:
```
np.float64(0.7603832562125579)
```

<!-- cell:15 type:markdown -->
You might try and find a **credible interval**. This, unlike the wierd definition of confidence intervals, is exactly what you think it is, the amount of probability mass between certain percentages, like the middle 80%

<!-- cell:16 type:code -->
```python
np.percentile(samples, [10, 90])
```
Output:
```
array([0.44882814, 0.81333566])
```

<!-- cell:17 type:markdown -->
You can make various point estimates: mean, median

<!-- cell:18 type:code -->
```python
np.mean(samples), np.median(samples), np.percentile(samples, 50) #last 2 are same
```
Output:
```
(np.float64(0.6373160701823328),
 np.float64(0.6457855966822045),
 np.float64(0.6457855966822045))
```

<!-- cell:19 type:markdown -->
A particularly important and useful point estimate is the **MAP**, or "maximum a-posteriori" estimate, the value of the parameter at which the pdf (num-samples) reach a maximum.

<!-- cell:20 type:code -->
```python
sampleshisto = np.histogram(samples, bins=50)
```

<!-- cell:21 type:code -->
```python
maxcountindex = np.argmax(sampleshisto[0])
mapvalue = sampleshisto[1][maxcountindex]
print(maxcountindex, mapvalue)
```
Output:
```
29 0.642838008998395
```

<!-- cell:22 type:markdown -->
A principled way to get these point estimates is a **loss function**. This is the subject of decision theory, and we shall come to it soon. Different losses correspond to different well known point estimates, as we shall see.

But as a quick idea of this,  consider the squared error decision loss:

$$R(t) = E_{p(\theta \vert D)}[(\theta -t)^2] = \int d\theta  (\theta -t)^2  p(\theta \vert D)$$

$$\frac{dR(t)}{dt} = 0 \implies  \int  d\theta -2(\theta -t)p(\theta \vert D) = 0$$

or 

$$ t= \int d\theta \theta\,p(\theta \vert D) $$

or the mean of the posterior.

We can see this with some quick computation:

<!-- cell:23 type:code -->
```python
mse = [np.mean((xi-samples)**2) for xi in x]
plt.plot(x, mse);
print("Mean",np.mean(samples));
```
Output:
```
Mean 0.6373160701823328
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-13-output-2.png)

<!-- cell:24 type:markdown -->
## Obtaining the posterior predictive

Its easy to sample from any one probability to get the sampling distribution at a particular $\theta$

<!-- cell:25 type:code -->
```python
point3samps = np.random.binomial( len(data), 0.3, size=10000);
point7samps = np.random.binomial( len(data), 0.7, size=10000);
plt.hist(point3samps, lw=3, alpha=0.5, histtype="stepfilled", bins=np.arange(11));
plt.hist(point7samps, lw=3, alpha=0.3,histtype="stepfilled", bins=np.arange(11));
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-14-output-1.png)

<!-- cell:26 type:markdown -->
The posterior predictive:

$$p(y^{*} \vert D) = \int d\theta p(y^{*} \vert \theta) p(\theta \vert D)$$

seems to be a complex integral.  But if you parse it, its not so complex. This diagram from McElreath helps:

![The posterior predictive distribution as a mixture: each parameter value implies a sampling distribution, weighted by the posterior probability, producing the marginal prediction. From McElreath, Statistical Rethinking.](https://rahuldave.com/posts/globemodel/assets/postpred.png)

A similar risk-minimization holds for the posterior-predictive  so that

$$y_{min mse} = \int  dy \, y \, p(y \vert D)$$

which is indeed what we  would use in a regression scenario...


### Plug-in Approximation

Also, often, people will use the **plug-in approximation** by putting the posterior mean or MAP value 

$$p(\theta \vert D) = \delta(\theta - \theta_{MAP})$$

and then simply  drawing the posterior predictive  from :

$$p(y^{*} \vert D) = p(y^{*} \vert \theta_{MAP})$$

(the same thing could be done for $\theta_{mean}$).

<!-- cell:27 type:code -->
```python
pluginpreds = np.random.binomial( len(data), mapvalue, size = len(samples))
```

<!-- cell:28 type:code -->
```python
plt.hist(pluginpreds, bins=np.arange(11));
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-16-output-1.png)

<!-- cell:29 type:markdown -->
This approximation is just sampling from the likelihood(sampling distribution), at a posterior-obtained value of $\theta$.  It might be useful if the posterior is an expensive MCMC and the MAP is easier to find by optimization, and can be used in conjunction with quadratic (gaussian) approximations to the posterior, as we will see in variational inference. But for now we have all the samples, and it would be inane not to use them...

<!-- cell:30 type:markdown -->
### The posterior predictive from sampling

But really from the perspective of sampling, all we have to do is to first draw the thetas from the posterior, then draw y's from the likelihood, and histogram the likelihood. This is the same logic as marginal posteriors, with the addition of the fact that we must draw  y from the likelihood once we drew $\theta$. You might think that we have to draw multiple $y$s at a theta, but this is already taken care of for us because of the nature of sampling. We already have multiple $\theta$a in a bin.

<!-- cell:31 type:code -->
```python
postpred = np.random.binomial( len(data), samples);
```

<!-- cell:32 type:code -->
```python
postpred
```
Output:
```
array([7, 8, 8, ..., 5, 6, 8], shape=(10000,))
```

<!-- cell:33 type:code -->
```python
samples.shape, postpred.shape
```
Output:
```
((10000,), (10000,))
```

<!-- cell:34 type:code -->
```python
plt.hist(postpred, bins=np.arange(11), alpha=0.5, align="left", label="predictive")
plt.hist(pluginpreds, bins=np.arange(11), alpha=0.2, align="left", label="plug-in (MAP)")
plt.title('Posterior predictive')
plt.xlabel('k')
plt.legend()
```
![Figure](https://rahuldave.com/posts/globemodel/index_files/figure-html/cell-20-output-1.png)

<!-- cell:35 type:markdown -->
You can interrogate the posterior-predictive, or **simulated** samples in other ways, asking about the longest run of water tosses, or the number of times the water/land switched. This is left as an exercise. In particular, you will find that the number of switches is not consistent with what you see in our data. This might lead you to question our model...always a good thing..but note that we have very little data as yet to go on
