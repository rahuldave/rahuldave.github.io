<!-- cell:1 type:code -->
```python
# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
%matplotlib inline 
#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
```

<!-- cell:2 type:markdown -->
Let us get the data and put it into a dataframe.

<!-- cell:3 type:code -->
```python
placebo = [54, 51, 58, 44, 55, 52, 42, 47, 58, 46]
drug = [54, 73, 53, 70, 73, 68, 52, 65, 65]
dosage = placebo + drug
label = ['P']*len(placebo) + ['D']*len(drug)
df = pd.DataFrame(dict(dosage=dosage, label=label))
df
```
Output:
```
    dosage label
0       54     P
1       51     P
2       58     P
3       44     P
4       55     P
5       52     P
6       42     P
7       47     P
8       58     P
9       46     P
10      54     D
11      73     D
12      53     D
13      70     D
14      73     D
15      68     D
16      52     D
17      65     D
18      65     D
```

<!-- cell:4 type:markdown -->
The "mean" size of the effect in our sample is about 13.

<!-- cell:5 type:code -->
```python
actuals = df.groupby('label').dosage.mean()
actuals
```
Output:
```
label
D    63.666667
P    50.700000
Name: dosage, dtype: float64
```

<!-- cell:6 type:code -->
```python
df.groupby('label').dosage.hist(bins=np.arange(30, 80, 1));
```
[Figure]

<!-- cell:7 type:code -->
```python
actual_effect = actuals['D'] - actuals['P']
actual_effect
```
Output:
```
12.966666666666661
```

<!-- cell:8 type:markdown -->
## Permutations to get significance

*Could it have happened by chance?*

We permute, group-by labels again, and calculate the effect. This kind of randomization should "kill" the effect:

<!-- cell:9 type:code -->
```python
temp = np.random.permutation(df.label)
```

<!-- cell:10 type:code -->
```python
temp_series = df.groupby(temp).dosage.mean()
temp_series
```
Output:
```
D    57.0
P    56.7
Name: dosage, dtype: float64
```

<!-- cell:11 type:code -->
```python
temp_series['D'] - temp_series['P']
```
Output:
```
0.29999999999999716
```

<!-- cell:12 type:markdown -->
If we compare the distribution of effect sizes to the actual effect, this actual effect should be in a tail if it is significant...

<!-- cell:13 type:code -->
```python
sig_means = np.zeros(10000)
for i in range(10000):
    temp = np.random.permutation(df.label)
    mean_series = df.groupby(temp).dosage.mean()
    sig_means[i] = mean_series['D'] - mean_series['P']

```

<!-- cell:14 type:code -->
```python
plt.hist(sig_means, bins=50, alpha=0.4);
plt.axvline(actual_effect, 0, 1, color="red");
```
[Figure]

<!-- cell:15 type:markdown -->
As a comparison, consider the case in which placebos had a much wider spread, between 50, and 450. Simply add 13 to each placebo value to get a dosage value. The mean difference would still be 13. But now, 13 would be way inside the histogram, and the effect would not be a significant one, and could have happened by chance.

<!-- cell:16 type:markdown -->
*Statistically significant does not mean important*. Thats a question of, how large is the effect, or where are the confidence intervals for the effect. For instance, if a statistically significant increase in mortality was a mean of 5 days over 5 years by drug over placebo, you would not consider the effect important.

<!-- cell:17 type:markdown -->
## Bootstrap to estimate size of effect

Here we randomize labels within the group, take means, and subtract. Here is an example

<!-- cell:18 type:code -->
```python
placebo_bs = np.random.choice(list(range(10)), size=(10000, 10))
drug_bs = np.random.choice(list(range(10, 19)), size=(10000, 9))
```

<!-- cell:19 type:code -->
```python
placebo_bs[0,:]
```
Output:
```
array([7, 7, 1, 5, 1, 5, 4, 6, 0, 7])
```

<!-- cell:20 type:code -->
```python
df.iloc[placebo_bs[0,:]]
```
Output:
```
   dosage label
7      47     P
7      47     P
1      51     P
5      52     P
1      51     P
5      52     P
4      55     P
6      42     P
0      54     P
7      47     P
```

<!-- cell:21 type:markdown -->
Here is the effect:

<!-- cell:22 type:code -->
```python
df.iloc[drug_bs[0,:]].dosage.mean() - df.iloc[placebo_bs[0,:]].dosage.mean()
```
Output:
```
14.977777777777774
```

<!-- cell:23 type:markdown -->
Let us do this 10000 times.

<!-- cell:24 type:code -->
```python
effect_diffs = np.zeros(10000)
for i in range(10000):
    effect_diffs[i] = df.iloc[drug_bs[i,:]].dosage.mean() - df.iloc[placebo_bs[i,:]].dosage.mean()
```

<!-- cell:25 type:code -->
```python
percs = np.percentile(effect_diffs, [5, 50, 95])
percs
```
Output:
```
array([  7.53333333,  13.05      ,  18.12222222])
```

<!-- cell:26 type:code -->
```python
plt.hist(effect_diffs, bins=100, alpha=0.2);
plt.axvline(actual_effect, 0, 1, color="red");
for p in percs:
    plt.axvline(p, 0, 1, color="green");
```
[Figure]

<!-- cell:27 type:markdown -->
That is, 90% of the time, the drug is 7.53 to 18.12 more effective than placebo. The average value of placebo in our sample was 50. This makes the drug 13 to 33% more effective, roghly, which seems it might be an important effect.

<!-- cell:28 type:markdown -->
If you have such a confidence interval, why do a significance test. Consider the extreme case of 2 data points, wel separated. The confidence interval is tight around the difference. But a permutation test would show that half the time, you will by random chance, get a difference just as big as  the observed one. Intuitively this is too little data to show significance, and this "half the time" bears that out...
