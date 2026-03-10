<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pymc",
#   "scikit-learn",
#   "seaborn",
# ]
# ///

```

<!-- cell:2 type:code -->
```python
%matplotlib inline
import pytensor
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
sns.set_style('whitegrid')
```

<!-- cell:3 type:code -->
```python
X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
```

<!-- cell:4 type:code -->
```python
Y = Y.astype('float64')
```

<!-- cell:5 type:code -->
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
```

<!-- cell:6 type:code -->
```python
fig, ax = plt.subplots()
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-7-output-1.png)

<!-- cell:7 type:code -->
```python
X_train.shape
```
Output:
```
(500, 2)
```

<!-- cell:8 type:code -->
```python
def construct_nn(ann_input, ann_output):
    n_hidden = 5

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(np.float64)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(np.float64)
    init_out = np.random.randn(n_hidden).astype(np.float64)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sigma=1,
                                 shape=(X.shape[1], n_hidden),
                                 initval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sigma=1,
                                shape=(n_hidden, n_hidden),
                                initval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sigma=1,
                                  shape=(n_hidden,),
                                  initval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input,
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1,
                                         weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2,
                                              weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out',
                           p=act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc will just use the new data.
# Kind-of like a pointer we can redirect.
ann_input = pytensor.shared(X_train)
ann_output = pytensor.shared(Y_train)
neural_network = construct_nn(ann_input, ann_output)
```

<!-- cell:9 type:code -->
```python
with neural_network:
    nutstrace = pm.sample(2000, tune=1000)
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
NUTS: [w_in_1, w_1_2, w_2_out]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/wV-uT_3pb4u247-POgKgx/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 86 seconds.
```
Output:
```
There were 1324 divergences after tuning. Increase `target_accept` or reparameterize.
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
az.summary(nutstrace)
```
Output:
```
               mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  \
w_in_1[0, 0] -0.252  1.517  -3.140    2.490      0.235    0.076      48.0   
w_in_1[0, 1]  0.491  1.602  -2.438    3.289      0.337    0.083      25.0   
w_in_1[0, 2] -0.230  1.559  -3.060    2.722      0.174    0.136      85.0   
w_in_1[0, 3] -0.006  1.327  -2.855    2.509      0.134    0.083     105.0   
w_in_1[0, 4] -0.067  1.507  -2.828    2.824      0.152    0.088     107.0   
w_in_1[1, 0]  0.095  0.673  -1.151    1.740      0.063    0.043     112.0   
w_in_1[1, 1]  0.109  0.670  -1.344    1.489      0.088    0.045      51.0   
w_in_1[1, 2] -0.025  0.575  -1.255    1.280      0.042    0.035     182.0   
w_in_1[1, 3] -0.047  0.607  -1.451    1.246      0.035    0.037     295.0   
w_in_1[1, 4] -0.020  0.623  -1.367    1.339      0.037    0.046     268.0   
w_1_2[0, 0]   0.018  1.249  -2.313    2.341      0.046    0.021     759.0   
w_1_2[0, 1]  -0.109  1.266  -2.463    2.251      0.043    0.025     878.0   
w_1_2[0, 2]   0.041  1.244  -2.286    2.308      0.039    0.021    1024.0   
w_1_2[0, 3]   0.012  1.259  -2.227    2.436      0.039    0.022    1076.0   
w_1_2[0, 4]   0.018  1.249  -2.236    2.403      0.038    0.023    1075.0   
w_1_2[1, 0]  -0.088  1.237  -2.289    2.282      0.046    0.020     741.0   
w_1_2[1, 1]  -0.037  1.273  -2.374    2.390      0.067    0.037     361.0   
w_1_2[1, 2]   0.033  1.223  -2.330    2.216      0.039    0.019    1007.0   
w_1_2[1, 3]   0.085  1.266  -2.222    2.406      0.043    0.020     899.0   
w_1_2[1, 4]   0.012  1.232  -2.252    2.269      0.040    0.021     948.0   
w_1_2[2, 0]  -0.054  1.300  -2.407    2.345      0.061    0.025     457.0   
w_1_2[2, 1]  -0.064  1.285  -2.365    2.383      0.065    0.022     400.0   
w_1_2[2, 2]   0.030  1.248  -2.287    2.306      0.041    0.018     950.0   
w_1_2[2, 3]  -0.022  1.273  -2.355    2.349      0.046    0.024     771.0   
w_1_2[2, 4]  -0.042  1.278  -2.354    2.400      0.060    0.046     456.0   
w_1_2[3, 0]   0.083  1.269  -2.274    2.491      0.047    0.023     719.0   
w_1_2[3, 1]   0.176  1.352  -2.329    2.724      0.076    0.042     321.0   
w_1_2[3, 2]   0.020  1.235  -2.304    2.305      0.043    0.021     852.0   
w_1_2[3, 3]  -0.020  1.257  -2.279    2.358      0.049    0.026     670.0   
w_1_2[3, 4]   0.043  1.293  -2.244    2.467      0.051    0.027     647.0   
w_1_2[4, 0]   0.042  1.268  -2.253    2.453      0.052    0.024     601.0   
w_1_2[4, 1]   0.090  1.310  -2.536    2.390      0.074    0.041     314.0   
w_1_2[4, 2]  -0.010  1.245  -2.368    2.297      0.041    0.022     924.0   
w_1_2[4, 3]  -0.042  1.271  -2.310    2.371      0.046    0.024     771.0   
w_1_2[4, 4]  -0.013  1.274  -2.448    2.240      0.051    0.022     637.0   
w_2_out[0]    0.254  2.365  -4.302    4.399      0.125    0.049     374.0   
w_2_out[1]    0.161  2.468  -4.633    4.412      0.176    0.069     196.0   
w_2_out[2]   -0.089  2.228  -4.237    4.191      0.083    0.045     736.0   
w_2_out[3]    0.177  2.365  -4.232    4.607      0.128    0.058     353.0   
w_2_out[4]   -0.101  2.346  -4.494    4.114      0.113    0.050     448.0   

              ess_tail  r_hat  
w_in_1[0, 0]     243.0   1.09  
w_in_1[0, 1]     203.0   1.11  
w_in_1[0, 2]     167.0   1.03  
w_in_1[0, 3]      90.0   1.03  
w_in_1[0, 4]     273.0   1.05  
w_in_1[1, 0]     223.0   1.03  
w_in_1[1, 1]     192.0   1.07  
w_in_1[1, 2]     280.0   1.02  
w_in_1[1, 3]     348.0   1.03  
w_in_1[1, 4]     387.0   1.03  
w_1_2[0, 0]     1966.0   1.01  
w_1_2[0, 1]     1319.0   1.01  
w_1_2[0, 2]     1661.0   1.00  
w_1_2[0, 3]     2152.0   1.01  
w_1_2[0, 4]     1865.0   1.00  
w_1_2[1, 0]     1930.0   1.01  
w_1_2[1, 1]      428.0   1.01  
w_1_2[1, 2]     1920.0   1.00  
w_1_2[1, 3]     2093.0   1.00  
w_1_2[1, 4]     2108.0   1.01  
w_1_2[2, 0]     1600.0   1.00  
w_1_2[2, 1]     1204.0   1.01  
w_1_2[2, 2]     2124.0   1.00  
w_1_2[2, 3]     1048.0   1.00  
w_1_2[2, 4]      196.0   1.01  
w_1_2[3, 0]     1048.0   1.01  
w_1_2[3, 1]      497.0   1.01  
w_1_2[3, 2]     1649.0   1.01  
w_1_2[3, 3]     1065.0   1.01  
w_1_2[3, 4]     1191.0   1.00  
w_1_2[4, 0]     1333.0   1.01  
w_1_2[4, 1]      482.0   1.02  
w_1_2[4, 2]     1141.0   1.00  
w_1_2[4, 3]     1208.0   1.00  
w_1_2[4, 4]      775.0   1.00  
w_2_out[0]       807.0   1.01  
w_2_out[1]       627.0   1.01  
w_2_out[2]      1540.0   1.00  
w_2_out[3]      1147.0   1.01  
w_2_out[4]      1317.0   1.00
```

<!-- cell:11 type:code -->
```python
with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=30000, method=inference)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/wV-uT_3pb4u247-POgKgx/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 170.33
```

<!-- cell:12 type:code -->
```python
advitrace = approx.sample(draws=5000)
```

<!-- cell:13 type:code -->
```python
az.summary(advitrace)
```
Output:
```
arviz - WARNING - Shape validation failed: input_shape: (1, 5000), minimum_shape: (chains=2, draws=4)
```
Output:
```
               mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  \
w_in_1[0, 0]  0.205  0.392  -0.539    0.904      0.006    0.004    4488.0   
w_in_1[0, 1]  0.545  0.136   0.285    0.791      0.002    0.001    4718.0   
w_in_1[0, 2] -2.102  0.556  -3.149   -1.092      0.008    0.006    4495.0   
w_in_1[0, 3]  0.560  0.132   0.304    0.804      0.002    0.001    5250.0   
w_in_1[0, 4] -0.208  0.527  -1.169    0.809      0.007    0.005    4996.0   
w_in_1[1, 0] -0.246  0.531  -1.229    0.789      0.008    0.005    4596.0   
w_in_1[1, 1] -0.478  0.255  -0.939    0.005      0.004    0.003    4909.0   
w_in_1[1, 2] -1.276  0.527  -2.258   -0.263      0.008    0.005    4894.0   
w_in_1[1, 3] -0.490  0.214  -0.889   -0.081      0.003    0.002    4709.0   
w_in_1[1, 4]  0.390  0.637  -0.821    1.552      0.009    0.007    4960.0   
w_1_2[0, 0]   0.007  0.995  -1.957    1.790      0.014    0.010    4931.0   
w_1_2[0, 1]   0.174  1.041  -1.708    2.144      0.015    0.011    4788.0   
w_1_2[0, 2]  -0.080  0.602  -1.180    1.070      0.009    0.006    4700.0   
w_1_2[0, 3]  -0.134  0.603  -1.220    1.006      0.009    0.006    4679.0   
w_1_2[0, 4]   0.209  0.569  -0.830    1.321      0.009    0.006    4435.0   
w_1_2[1, 0]  -0.315  0.943  -2.119    1.369      0.014    0.009    4703.0   
w_1_2[1, 1]   0.330  0.972  -1.509    2.145      0.014    0.010    5166.0   
w_1_2[1, 2]  -1.088  0.537  -2.103   -0.095      0.008    0.005    4404.0   
w_1_2[1, 3]  -0.888  0.550  -1.935    0.138      0.008    0.005    5042.0   
w_1_2[1, 4]   1.284  0.496   0.358    2.215      0.007    0.005    5161.0   
w_1_2[2, 0]  -0.485  0.844  -2.117    1.039      0.012    0.009    4709.0   
w_1_2[2, 1]   0.444  0.871  -1.180    2.125      0.012    0.009    4919.0   
w_1_2[2, 2]  -0.730  0.359  -1.387   -0.034      0.005    0.004    4661.0   
w_1_2[2, 3]  -0.785  0.357  -1.455   -0.101      0.005    0.004    4994.0   
w_1_2[2, 4]   1.000  0.348   0.317    1.619      0.005    0.003    4949.0   
w_1_2[3, 0]  -0.295  0.944  -2.071    1.439      0.014    0.010    4820.0   
w_1_2[3, 1]   0.045  0.979  -1.757    1.873      0.014    0.010    4854.0   
w_1_2[3, 2]  -1.151  0.533  -2.129   -0.128      0.008    0.005    5021.0   
w_1_2[3, 3]  -1.384  0.528  -2.420   -0.405      0.008    0.005    4921.0   
w_1_2[3, 4]   1.139  0.485   0.201    2.029      0.007    0.005    4924.0   
w_1_2[4, 0]   0.247  0.962  -1.606    2.041      0.014    0.010    4792.0   
w_1_2[4, 1]  -0.207  0.973  -2.013    1.607      0.014    0.010    4962.0   
w_1_2[4, 2]   0.176  0.531  -0.811    1.189      0.008    0.005    4972.0   
w_1_2[4, 3]   0.261  0.554  -0.815    1.270      0.008    0.005    4788.0   
w_1_2[4, 4]   0.154  0.501  -0.755    1.104      0.007    0.005    5008.0   
w_2_out[0]   -0.193  0.244  -0.649    0.258      0.003    0.003    5080.0   
w_2_out[1]    0.147  0.245  -0.308    0.606      0.003    0.002    5096.0   
w_2_out[2]   -1.140  0.231  -1.580   -0.706      0.003    0.002    5087.0   
w_2_out[3]   -1.148  0.236  -1.589   -0.709      0.003    0.002    5020.0   
w_2_out[4]    1.211  0.232   0.778    1.631      0.003    0.002    5120.0   

              ess_tail  r_hat  
w_in_1[0, 0]    5023.0    NaN  
w_in_1[0, 1]    4809.0    NaN  
w_in_1[0, 2]    4604.0    NaN  
w_in_1[0, 3]    4973.0    NaN  
w_in_1[0, 4]    4938.0    NaN  
w_in_1[1, 0]    4611.0    NaN  
w_in_1[1, 1]    4759.0    NaN  
w_in_1[1, 2]    4982.0    NaN  
w_in_1[1, 3]    4972.0    NaN  
w_in_1[1, 4]    4601.0    NaN  
w_1_2[0, 0]     4778.0    NaN  
w_1_2[0, 1]     4808.0    NaN  
w_1_2[0, 2]     4935.0    NaN  
w_1_2[0, 3]     4726.0    NaN  
w_1_2[0, 4]     4906.0    NaN  
w_1_2[1, 0]     4713.0    NaN  
w_1_2[1, 1]     4631.0    NaN  
w_1_2[1, 2]     4605.0    NaN  
w_1_2[1, 3]     4912.0    NaN  
w_1_2[1, 4]     4813.0    NaN  
w_1_2[2, 0]     4754.0    NaN  
w_1_2[2, 1]     4782.0    NaN  
w_1_2[2, 2]     4599.0    NaN  
w_1_2[2, 3]     5021.0    NaN  
w_1_2[2, 4]     4941.0    NaN  
w_1_2[3, 0]     4909.0    NaN  
w_1_2[3, 1]     4590.0    NaN  
w_1_2[3, 2]     5017.0    NaN  
w_1_2[3, 3]     4901.0    NaN  
w_1_2[3, 4]     4938.0    NaN  
w_1_2[4, 0]     4866.0    NaN  
w_1_2[4, 1]     4785.0    NaN  
w_1_2[4, 2]     4812.0    NaN  
w_1_2[4, 3]     4494.0    NaN  
w_1_2[4, 4]     4458.0    NaN  
w_2_out[0]      4896.0    NaN  
w_2_out[1]      4971.0    NaN  
w_2_out[2]      4785.0    NaN  
w_2_out[3]      5066.0    NaN  
w_2_out[4]      4901.0    NaN
```

<!-- cell:14 type:code -->
```python
plt.plot(-inference.hist, alpha=.3)
```
Output:
```
[<matplotlib.lines.Line2D at 0x116375940>]
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-15-output-1.png)

<!-- cell:15 type:code -->
```python
ann_input.set_value(X_test)
ann_output.set_value(Y_test)
with neural_network:
    ppc = pm.sample_posterior_predictive(advitrace)
```
Output:
```
Sampling: [out]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/wV-uT_3pb4u247-POgKgx/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:16 type:code -->
```python
pred = ppc.posterior_predictive['out'].values.reshape(-1, X_test.shape[0]).mean(axis=0) > 0.5
```

<!-- cell:17 type:code -->
```python
fig, ax = plt.subplots()
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
sns.despine()
ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-18-output-1.png)

<!-- cell:18 type:code -->
```python
print('Accuracy = {}%'.format((Y_test == pred).mean() * 100))
```
Output:
```
Accuracy = 88.8%
```

<!-- cell:19 type:code -->
```python
grid = pm.floatX(np.mgrid[-3:3:100j,-3:3:100j])
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid_2d.shape[0], dtype=np.float64)
```

<!-- cell:20 type:code -->
```python
ann_input.set_value(grid_2d)
ann_output.set_value(dummy_out)
with neural_network:
    ppc_grid = pm.sample_posterior_predictive(advitrace, var_names=['out'])
```
Output:
```
Sampling: [out]
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/wV-uT_3pb4u247-POgKgx/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```

<!-- cell:21 type:code -->
```python
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
fig, ax = plt.subplots(figsize=(16, 9))
ppc_out_mean = ppc_grid.posterior_predictive['out'].values.reshape(-1, grid_2d.shape[0]).mean(axis=0)
contour = ax.contourf(grid[0], grid[1], ppc_out_mean.reshape(100, 100), cmap=cmap)
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
cbar = plt.colorbar(contour, ax=ax)
_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0');
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-22-output-1.png)

<!-- cell:22 type:code -->
```python
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
fig, ax = plt.subplots(figsize=(16, 9))
ppc_out_std = ppc_grid.posterior_predictive['out'].values.reshape(-1, grid_2d.shape[0]).std(axis=0)
contour = ax.contourf(grid[0], grid[1], ppc_out_std.reshape(100, 100), cmap=cmap)
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
cbar = plt.colorbar(contour, ax=ax)
_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-23-output-1.png)

<!-- cell:23 type:code -->
```python
minibatch_x = pm.Minibatch(X_train, batch_size=50)
minibatch_y = pm.Minibatch(Y_train, batch_size=50)
neural_network_minibatch = construct_nn(minibatch_x, minibatch_y)
with neural_network_minibatch:
    approx_mb = pm.fit(40000, method=pm.ADVI())
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/wV-uT_3pb4u247-POgKgx/lib/python3.14/site-packages/rich/live.py:260: 
UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
```
Output:
```
Finished [100%]: Average Loss = 35.717
```

<!-- cell:24 type:code -->
```python
plt.plot(-approx_mb.hist)
```
Output:
```
[<matplotlib.lines.Line2D at 0x115b89940>]
```
![Figure](https://rahuldave.com/posts/varnn/index_files/figure-html/cell-25-output-1.png)

<!-- cell:25 type:code -->
```python

```
