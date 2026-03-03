<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
#   "seaborn",
#   "statsmodels",
# ]
# ///

```

<!-- cell:2 type:markdown -->
This notebook is based on McElreath, Rethinking Statistics, Chapter 6.

<!-- cell:3 type:markdown -->
When we use the empirical distribution and sample quantities here we are working with our training sample (s).

Clearly we can calculate deviance on the validation and test samples as well to remedy this issue. And the results will be similar to what we found in lecture for MSE, with the training deviance decreasing with complexity and the testing deviance increasing at some point. 

<!-- cell:4 type:code -->
```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```

<!-- cell:5 type:markdown -->
## A trick to generate data

We generate data from a gaussian with standard deviation 1 and means given by:

$$\mu_i = 0.15 x_{1,i} - 0.4 x_{2,i}, y \sim N(\mu, 1).$$

This is a **2 parameter** model.

We use an interesting trick to generate this data, directly using the regression coefficients as correlations with the response variable.

<!-- cell:6 type:code -->
```python
def generate_data(N, k, rho=[0.15, -0.4]):
    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k
    Rho = np.eye(n_dim)
    for i,r in enumerate(rho):
        Rho[0, i+1] = r
    index_lower = np.tril_indices(n_dim, -1)
    Rho[index_lower] = Rho.T[index_lower]
    mean = n_dim * [0.]
    Xtrain = np.random.multivariate_normal(mean, Rho, size=N)
    Xtest = np.random.multivariate_normal(mean, Rho, size=N)
    ytrain = Xtrain[:,0].copy()
    Xtrain[:,0]=1.
    ytest = Xtest[:,0].copy()
    Xtest[:,0]=1.
    return Xtrain[:,:k], ytrain, Xtest[:,:k], ytest
```

<!-- cell:7 type:markdown -->
We want to generate data for 5 different cases, a one parameter (intercept) fit, a two parameter (intercept and $x_1$), three parameters (add a $x_2), and four and five parameters. Here is what the data looks like for 2 parameters:

<!-- cell:8 type:code -->
```python
generate_data(20,2)
```
Output:
```
(array([[ 1.        , -1.85105554],
        [ 1.        ,  0.67309992],
        [ 1.        ,  0.34594768],
        [ 1.        ,  0.65064179],
        [ 1.        ,  0.68646577],
        [ 1.        ,  1.04516541],
        [ 1.        ,  0.25350865],
        [ 1.        , -1.75025994],
        [ 1.        ,  2.04070932],
        [ 1.        ,  0.20222876],
        [ 1.        , -0.50638673],
        [ 1.        ,  0.0935065 ],
        [ 1.        , -1.25684132],
        [ 1.        ,  0.04554159],
        [ 1.        , -0.33088164],
        [ 1.        ,  0.32078227],
        [ 1.        ,  0.38402417],
        [ 1.        , -1.01503021],
        [ 1.        , -1.40990038],
        [ 1.        ,  0.5689682 ]]),
 array([-0.03094429,  2.47155808, -0.42574492,  2.25816724,  0.62396049,
        -0.8549415 ,  0.55872037,  0.4524301 ,  1.79346019, -1.17620102,
         0.44646631, -0.70015394,  0.1351058 , -1.41284706,  0.58926921,
        -0.38128767,  1.1317747 , -0.66355193,  0.76844869, -0.47215022]),
 array([[ 1.        , -0.81810994],
        [ 1.        ,  0.29573776],
        [ 1.        ,  0.6267238 ],
        [ 1.        ,  0.57358337],
        [ 1.        , -2.26842333],
        [ 1.        ,  1.32234749],
        [ 1.        , -1.2340754 ],
        [ 1.        ,  0.44374861],
        [ 1.        ,  0.4587036 ],
        [ 1.        ,  2.10919308],
        [ 1.        ,  1.79686875],
        [ 1.        , -0.70098456],
        [ 1.        ,  0.88812631],
        [ 1.        ,  0.63059487],
        [ 1.        ,  2.12570521],
        [ 1.        ,  0.94311861],
        [ 1.        , -0.83029288],
        [ 1.        , -0.39032711],
        [ 1.        ,  1.32121167],
        [ 1.        ,  1.53094961]]),
 array([-0.21283452,  1.57834009,  2.21346054, -0.15431662, -1.58394284,
         0.20391388,  0.60082183, -0.7488139 , -0.25319313, -1.14016847,
         1.28394113, -2.08515239, -0.41085624,  0.79644902,  0.53668242,
         1.40827029,  0.09794264,  0.1043875 ,  0.16872142,  0.14252517]))
```

<!-- cell:9 type:markdown -->
And for four parameters

<!-- cell:10 type:code -->
```python
generate_data(20,4)
```
Output:
```
(array([[ 1.        , -0.29277006, -0.733162  , -1.39574915],
        [ 1.        ,  0.61770683, -0.51010952, -0.45268044],
        [ 1.        ,  0.12326025,  0.45269595, -0.47725029],
        [ 1.        ,  0.21423856,  0.87942464, -0.24379707],
        [ 1.        ,  1.06791532, -0.15554897, -0.96213956],
        [ 1.        ,  0.66914579, -0.30641499, -0.18514654],
        [ 1.        ,  0.51792011,  2.38742869,  1.6403617 ],
        [ 1.        ,  1.32286386, -0.53007569,  0.56252102],
        [ 1.        , -1.45203704, -2.30382682, -0.85244161],
        [ 1.        ,  0.41303584,  0.37374287, -0.65564207],
        [ 1.        ,  0.32718023, -0.01749096,  0.95469287],
        [ 1.        , -0.62450428, -0.58733704, -1.3872151 ],
        [ 1.        ,  0.00685922,  0.55999523,  0.79231594],
        [ 1.        , -0.42154584, -0.6048118 ,  0.75780672],
        [ 1.        ,  1.45059015, -1.31438536, -1.65630614],
        [ 1.        , -0.03938478, -0.44487804, -0.1961614 ],
        [ 1.        ,  1.36566779, -0.64068386, -0.54875285],
        [ 1.        ,  1.72482309,  1.53040087, -0.26647265],
        [ 1.        ,  0.23220957, -2.20049614,  1.78430743],
        [ 1.        , -0.00670065, -0.32290568,  1.79839903]]),
 array([ 0.39498688,  1.87643175,  0.94510903, -0.8066848 ,  0.19627142,
        -0.10908993, -1.27014707, -0.08417742,  1.82719932,  0.35594856,
         0.63598407, -1.2661829 , -0.59666786,  2.65137374, -1.18213587,
        -1.13867314, -0.20497472,  0.62803383,  0.35626044, -0.67891133]),
 array([[ 1.        , -0.57065779, -0.51324083,  0.10670859],
        [ 1.        ,  1.0087654 ,  0.38755674,  1.67852244],
        [ 1.        , -0.49429559, -0.57041062,  1.56430826],
        [ 1.        , -1.45855266,  0.1558655 , -0.38637072],
        [ 1.        ,  1.03070414, -0.48417865,  0.8745302 ],
        [ 1.        , -0.40188239, -1.41209873, -0.10114362],
        [ 1.        , -0.47092572,  0.87573497,  1.5544465 ],
        [ 1.        ,  0.44560682,  2.20104727,  0.52994293],
        [ 1.        ,  0.43706979, -0.32462082,  1.33187328],
        [ 1.        , -0.12743926,  0.15745755,  1.88201746],
        [ 1.        , -1.41859498, -1.53000908,  1.0654683 ],
        [ 1.        ,  1.07383216, -0.68657856,  0.02585957],
        [ 1.        , -1.64467412,  1.29249625,  1.09050156],
        [ 1.        , -0.32409208, -1.24634146,  0.30853506],
        [ 1.        , -0.77407105, -0.45128643,  0.18307768],
        [ 1.        , -0.13901513,  0.84551355, -1.55339667],
        [ 1.        , -0.8304255 , -0.34530494, -0.29643615],
        [ 1.        ,  0.37648098,  0.59186511, -1.10515656],
        [ 1.        ,  0.00237547, -0.49621492,  0.32752447],
        [ 1.        ,  0.60316126,  0.01008026,  0.70366708]]),
 array([-1.43661308,  0.00590898, -1.68712387, -0.46413583, -2.22275277,
         1.05062162, -2.23715934,  0.08897373, -0.3568274 , -0.85962375,
         0.7131029 ,  2.2103827 , -0.47798409,  0.89204121, -1.52284713,
        -1.24195911,  0.50554727, -0.143756  ,  0.34538576, -0.3707892 ]))
```

<!-- cell:11 type:code -->
```python
from scipy.stats import norm
import statsmodels.api as sm
```

<!-- cell:12 type:markdown -->
## Analysis, n=20

Here is the main loop of our analysis. We take the 5 models we talked about. For each model we generate 10000 samples of the data, split into an equal sized (N=20 each) training and testing set. We fit the regression on the training set, and calculate the deviance on the training set. Notice how we have simply used the `logpdf` from `scipy.stats`. You can easily do this for other distributions.

We then use the fit to calculate the $\mu$ on the test set, and calculate the deviance there. We then find the average and the standard deviation across the 10000 simulations.

Why do we do 10000 simulations? These are our **multiple samples from some hypothetical population**.

<!-- cell:13 type:code -->
```python
reps=10000
results_20 = {}
for k in range(1,6):
    trdevs=np.zeros(reps)
    tedevs=np.zeros(reps)
    for r in range(reps):
        Xtr, ytr, Xte, yte = generate_data(20, k)
        ols = sm.OLS(ytr, Xtr).fit()
        mutr = np.dot(Xtr, ols.params)
        devtr = -2*np.sum(norm.logpdf(ytr, mutr, 1))
        mute = np.dot(Xte, ols.params)
        #print(mutr.shape, mute.shape)
        devte = -2*np.sum(norm.logpdf(yte, mute, 1))
        #print(k, r, devtr, devte)
        trdevs[r] = devtr
        tedevs[r] = devte
    results_20[k] = (np.mean(trdevs), np.std(trdevs), np.mean(tedevs), np.std(tedevs))
```

<!-- cell:14 type:code -->
```python
import pandas as pd
df = pd.DataFrame(results_20).T
df = df.rename(columns = dict(zip(range(4), ['train', 'train_std', 'test', 'test_std'])))
df
```
Output:
```
       train  train_std       test  test_std
1  55.800925   6.232833  57.777482  6.797883
2  54.278719   5.876682  58.526214  7.369491
3  50.680767   4.785705  56.083845  6.809671
4  49.912570   4.632746  57.307597  7.575071
5  49.061402   4.513540  58.792026  8.564585
```

<!-- cell:15 type:code -->
```python
import seaborn as sns
colors = sns.color_palette()
colors
```
Output:
```
[(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
```

<!-- cell:16 type:markdown -->
We plot the traing and testing deviances

<!-- cell:17 type:code -->
```python
plt.plot(df.index, df.train, 'o', color = colors[0])
plt.errorbar(df.index, df.train, yerr=df.train_std, fmt='none', color=colors[0])
plt.plot(df.index+0.2, df.test, 'o', color = colors[1])
plt.errorbar(df.index+0.2, df.test, yerr=df.test_std, fmt='none', color=colors[1])
plt.xlabel("number of parameters")
plt.ylabel("deviance")
plt.title("N=20");
```
[Figure]

<!-- cell:18 type:markdown -->
Notice:

- the best fit model may not be the original generating model. Remember that the choice of fit depends on the amount of data you have and the less data you have, the less parameters you should use
- on average, out of sample deviance must be larger than in-sample deviance, through an individual pair may have that order reversed because of sample peculiarity.

## AIC, or the difference in deviances

Let us see the difference between the mean testing and training deviances. This is the difference in *bias* between the two sets.

<!-- cell:19 type:code -->
```python
df.test - df.train
```
Output:
```
1    1.976557
2    4.247494
3    5.403078
4    7.395027
5    9.730624
dtype: float64
```

<!-- cell:20 type:markdown -->
Voila, this seems to be roughly twice the number of parameters. In other words we might be able to get away without a test set if we "correct" the bias on the traing set by $2n_p$. This is the observation that motivates the AIC.

<!-- cell:21 type:markdown -->
### Analysis N=100

<!-- cell:22 type:code -->
```python
reps=10000
results_100 = {}
for k in range(1,6):
    trdevs=np.zeros(reps)
    tedevs=np.zeros(reps)
    for r in range(reps):
        Xtr, ytr, Xte, yte = generate_data(100, k)
        ols = sm.OLS(ytr, Xtr).fit()
        mutr = np.dot(Xtr, ols.params)
        devtr = -2*np.sum(norm.logpdf(ytr, mutr, 1))
        mute = np.dot(Xte, ols.params)
        devte = -2*np.sum(norm.logpdf(yte, mute, 1))
        #print(k, r, devtr, devte)
        trdevs[r] = devtr
        tedevs[r] = devte
    results_100[k] = (np.mean(trdevs), np.std(trdevs), np.mean(tedevs), np.std(tedevs))
```

<!-- cell:23 type:code -->
```python
df100 = pd.DataFrame(results_100).T
df100 = df100.rename(columns = dict(zip(range(4), ['train', 'train_std', 'test', 'test_std'])))
df100
```
Output:
```
        train  train_std        test   test_std
1  282.734005  14.014682  284.741736  14.358569
2  279.534664  13.678196  283.621105  13.998539
3  263.246835  11.202506  267.872399  11.930758
4  262.387224  11.421170  269.219596  12.440657
5  261.347208  11.246849  269.730672  12.551977
```

<!-- cell:24 type:code -->
```python
plt.plot(df100.index, df100.train, 'o', color = colors[0])
plt.errorbar(df100.index, df100.train, yerr=df100.train_std, fmt='none', color=colors[0])
plt.plot(df100.index+0.2, df100.test, 'o', color = colors[1])
plt.errorbar(df100.index+0.2, df100.test, yerr=df100.test_std, fmt='none', color=colors[1])
plt.xlabel("number of parameters")
plt.ylabel("deviance")
plt.title("N=100");
```
[Figure]

<!-- cell:25 type:code -->
```python
df100.test - df100.train
```
Output:
```
1    2.007731
2    4.086441
3    4.625564
4    6.832372
5    8.383464
dtype: float64
```

<!-- cell:26 type:markdown -->
We get pretty much the same result at N=100.

<!-- cell:27 type:markdown -->
## Assumptions for AIC

This observation leads to an estimate of the out-of-sample deviance by what is called an **information criterion**, the Akaike Information Criterion, or AIC:

$$AIC = D_{train} + 2n_p$$

which does carry as assumptions that

1. the likelihood is approximately multivariate gaussian
2. the sample size is much larger than the number of parameters
3. priors are flat 
4. The AIC does not assume that the true data generating process $p$ is in the set of models being fitted. The overarching goal of the AIC approach to model selection is to select the "best" model for our given data set without assuming that the "true" model is in the family of models from which we're selecting. The true model "cancels out" except in the expectation.

We wont derive the AIC here, but if you are interested, see  http://www.stat.cmu.edu/~larry/=stat705/Lecture16.pdf

Why would we want to use such information criteria? Cross validation can be expensive, especially with multiple hyper-parameters.

## AIC for Linear Regression

The AIC for a model is the training deviance plus twice the number of parameters:

$$AIC = D_{train} + 2n_p.$$

That is, -2 times the log likelihood of the model.

So, one we find the MLE solution for the linear regression, we plugin the values we get, which are

$$\sigma_{MLE}^2 =  \frac{1}{N} RSS $$

where RSS is the sum of the  squares of the errors.

$$AIC = -2(-\frac{N}{2}(log(2\pi) + log(\sigma^2)) -2(-\frac{1}{2\sigma_{MLE}^2} \times RSS) + 2p$$

Thus:

$$D = Nlog(RSS/N) $$

$$AIC = Nlog(RSS/N) + 2p + constant$$

Since the deviance for a OLS model is just proportional to the log(MSE) upto a proportionality, we'll use the MSE to derive this split.

The fact that the (log-likelihood) and thus the deviance  carries an expectation over the true distribution as estimated on the sample means that the **Deviance is a stochastic quantity, varying from sample to sample**.

<!-- cell:28 type:markdown -->
## A complete understanding of the comparison diagram

(taken from McElreath, but see upstairs as well)

![In-sample vs. out-of-sample deviance as model complexity increases, for N=20 and N=100. From McElreath, Statistical Rethinking.](assets/inoutdeviance.png)

<!-- cell:29 type:markdown -->
Now we are equipped to understand this diagram completely. Lets focus on the training (in) set first: blue points. 

1. There is some irreducible noise which contributes to the deviance no matter the number of parameters.
2. If we could capture the true model exactly there would be no **bias**, and the deviance would go to that which comes from the irreducible noise.
3. But we cant, so the positions of the circles tells us how much bias plus irreducible noise we have
4. The error bars now tell us our **variance**, since they tell us how much our deviance, or MSE varies around our "mean" model. In real life our sample will lie somewhere along this error bar.
5. The training set deviances go down as the number of parameters increase. The test set deviances go down and then go up
6. Notice that testing deviance is higher on a 2 parameter model than on a 1, even though our generating "true" model is a 2 parameter one. Deviance and the AIC do not pick the true model, but rather the one with the highest predictive accuracy.
