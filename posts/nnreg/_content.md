<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "torch",
# ]
# ///

```

<!-- cell:2 type:code -->
```python
import torch
import torch.nn as nn
from torch.nn import functional as fn
from torch.autograd import Variable
class MLRegP(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlinearity = fn.tanh, additional_hidden_wide=0):
        super(MLRegP, self).__init__()
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.fc_mid = nn.ModuleList()
        self.additional_hidden_wide = additional_hidden_wide
        for i in range(self.additional_hidden_wide):
            self.fc_mid.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.fc_initial(x)
        out_init = self.nonlinearity(x)
        x = self.nonlinearity(x)
        for i in range(self.additional_hidden_wide):
            x = self.fc_mid[i](x)
            x = self.nonlinearity(x)
        out_final = self.fc_final(x)
        return out_final, x, out_init
```

<!-- cell:3 type:code -->
```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 2*np.pi, 0.01)
y = np.sin(x) + 0.1*np.random.normal(size=x.shape[0])
xgrid=x
ygrid=y
plt.plot(x,y, '.', alpha=0.2);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-4-output-1.png)

<!-- cell:4 type:code -->
```python
xgrid.shape
```
Output:
```
(629,)
```

<!-- cell:5 type:code -->
```python
from sklearn.linear_model import LinearRegression
est = LinearRegression().fit(x.reshape(-1,1), y)
plt.plot(x,y, '.', alpha=0.2);
plt.plot(x,est.predict(x.reshape(-1,1)), 'k-', lw=3, alpha=0.2);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-6-output-1.png)

<!-- cell:6 type:code -->
```python
xdata = Variable(torch.Tensor(xgrid))
ydata = Variable(torch.Tensor(ygrid))
```

<!-- cell:7 type:code -->
```python
import torch.utils.data
dataset = torch.utils.data.TensorDataset(torch.from_numpy(xgrid.reshape(-1,1)), torch.from_numpy(ygrid))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```

<!-- cell:8 type:code -->
```python
dataset.tensors[0].shape, dataset.tensors[1].shape
```
Output:
```
(torch.Size([629, 1]), torch.Size([629]))
```

<!-- cell:9 type:code -->
```python
def run_model(model, epochs):
    criterion = nn.MSELoss()
    lr, epochs, batch_size = 1e-1 , epochs , 64
    optimizer = torch.optim.SGD(model.parameters(), lr = lr )
    accum=[]
    for k in range(epochs):
        localaccum = []
        for localx, localy in iter(loader):
            localx = Variable(localx.float())
            localy = Variable(localy.float())
            output, _, _ = model.forward(localx)
            loss = criterion(output, localy)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            localaccum.append(loss.item())
        accum.append((np.mean(localaccum), np.std(localaccum)))
    return accum
```

<!-- cell:10 type:markdown -->
### input dim 1, 2 hidden layers width 2, linear output

<!-- cell:11 type:code -->
```python
model = MLRegP(1, 2, nonlinearity=fn.sigmoid, additional_hidden_wide=1)
```

<!-- cell:12 type:code -->
```python
print(model)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=2, bias=True)
  (fc_mid): ModuleList(
    (0): Linear(in_features=2, out_features=2, bias=True)
  )
  (fc_final): Linear(in_features=2, out_features=1, bias=True)
)
```

<!-- cell:13 type:code -->
```python
accum = run_model(model, 2000)
```
Output:
```
/Users/rahul/Library/Caches/uv/archive-v0/vqGY-5rti2C5JP-dmfDIu/lib/python3.14/site-packages/torch/nn/modules/loss.py:626: UserWarning: Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
/Users/rahul/Library/Caches/uv/archive-v0/vqGY-5rti2C5JP-dmfDIu/lib/python3.14/site-packages/torch/nn/modules/loss.py:626: UserWarning: Using a target size (torch.Size([53])) that is different to the input size (torch.Size([53, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
```

<!-- cell:14 type:code -->
```python
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-14-output-1.png)

<!-- cell:15 type:code -->
```python
plt.plot([a[0]+a[1] for a in accum]);
plt.plot([a[0]-a[1] for a in accum]);
plt.xlim(0, 1000)
```
Output:
```
(0.0, 1000.0)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-15-output-1.png)

<!-- cell:16 type:code -->
```python
finaloutput, init_output, mid_output = model.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
#plt.xticks([])
#plt.yticks([])
```
Output:
```
[<matplotlib.lines.Line2D at 0x135036e40>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-16-output-1.png)

<!-- cell:17 type:code -->
```python
io = mid_output.data.numpy()
io.shape
```
Output:
```
(629, 2)
```

<!-- cell:18 type:code -->
```python
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-18-output-1.png)

<!-- cell:19 type:markdown -->
### input dim 1, 2 hidden layers width 4, linear output

<!-- cell:20 type:code -->
```python
model2 = MLRegP(1, 4, nonlinearity=fn.sigmoid, additional_hidden_wide=1)
accum = run_model(model2, 4000)
```

<!-- cell:21 type:code -->
```python
print(model2)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=4, bias=True)
  (fc_mid): ModuleList(
    (0): Linear(in_features=4, out_features=4, bias=True)
  )
  (fc_final): Linear(in_features=4, out_features=1, bias=True)
)
```

<!-- cell:22 type:code -->
```python
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-21-output-1.png)

<!-- cell:23 type:code -->
```python
finaloutput, init_output, mid_output = model2.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x134785e80>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-22-output-1.png)

<!-- cell:24 type:code -->
```python
io = mid_output.data.numpy()
io.shape
```
Output:
```
(629, 4)
```

<!-- cell:25 type:code -->
```python
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-24-output-1.png)

<!-- cell:26 type:markdown -->
### input dim 1, 2 hidden layers width 8, linear output

<!-- cell:27 type:code -->
```python
model3 = MLRegP(1, 8, nonlinearity=fn.sigmoid, additional_hidden_wide=1)
accum = run_model(model3, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-25-output-1.png)

<!-- cell:28 type:code -->
```python
print(model3)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=8, bias=True)
  (fc_mid): ModuleList(
    (0): Linear(in_features=8, out_features=8, bias=True)
  )
  (fc_final): Linear(in_features=8, out_features=1, bias=True)
)
```

<!-- cell:29 type:code -->
```python
finaloutput, init_output, mid_output = model3.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x135bf4d70>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-27-output-1.png)

<!-- cell:30 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-28-output-1.png)

<!-- cell:31 type:markdown -->
### input dim 1, 3 hidden layers width 4, linear output

<!-- cell:32 type:code -->
```python
model4 = MLRegP(1, 4, nonlinearity=fn.sigmoid, additional_hidden_wide=2)
accum = run_model(model4, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-29-output-1.png)

<!-- cell:33 type:code -->
```python
print(model4)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=4, bias=True)
  (fc_mid): ModuleList(
    (0-1): 2 x Linear(in_features=4, out_features=4, bias=True)
  )
  (fc_final): Linear(in_features=4, out_features=1, bias=True)
)
```

<!-- cell:34 type:code -->
```python
finaloutput, init_output, mid_output = model4.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x135d986e0>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-31-output-1.png)

<!-- cell:35 type:markdown -->
### input dim 1, 3 hidden layers width 2, linear output

<!-- cell:36 type:code -->
```python
model5 = MLRegP(1, 2, nonlinearity=fn.sigmoid, additional_hidden_wide=2)
accum = run_model(model5, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-32-output-1.png)

<!-- cell:37 type:code -->
```python
print(model5)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=2, bias=True)
  (fc_mid): ModuleList(
    (0-1): 2 x Linear(in_features=2, out_features=2, bias=True)
  )
  (fc_final): Linear(in_features=2, out_features=1, bias=True)
)
```

<!-- cell:38 type:code -->
```python
finaloutput, init_output, mid_output = model5.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x135e4de80>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-34-output-1.png)

<!-- cell:39 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-35-output-1.png)

<!-- cell:40 type:markdown -->
### input dim 1, 1 hidden layers width 2, linear output

<!-- cell:41 type:code -->
```python
model6 = MLRegP(1, 2, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model6, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-36-output-1.png)

<!-- cell:42 type:code -->
```python
print(model6)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=2, bias=True)
  (fc_mid): ModuleList()
  (fc_final): Linear(in_features=2, out_features=1, bias=True)
)
```

<!-- cell:43 type:code -->
```python
finaloutput, init_output, mid_output = model6.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x136808d70>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-38-output-1.png)

<!-- cell:44 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-39-output-1.png)

<!-- cell:45 type:markdown -->
### input dim 1, 1 hidden layers width 1, linear output

<!-- cell:46 type:code -->
```python
model7 = MLRegP(1, 1, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model7, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-40-output-1.png)

<!-- cell:47 type:code -->
```python
print(model7)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=1, bias=True)
  (fc_mid): ModuleList()
  (fc_final): Linear(in_features=1, out_features=1, bias=True)
)
```

<!-- cell:48 type:code -->
```python
finaloutput, init_output, mid_output = model7.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x135f7b0e0>]
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-42-output-1.png)

<!-- cell:49 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-43-output-1.png)

<!-- cell:50 type:markdown -->
### input dim 1, 1 hidden layers width 16, linear output

<!-- cell:51 type:code -->
```python
model8 = MLRegP(1, 16, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model8, 4000)
plt.plot([a[0] for a in accum]);
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-44-output-1.png)

<!-- cell:52 type:code -->
```python
print(model8)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=16, bias=True)
  (fc_mid): ModuleList()
  (fc_final): Linear(in_features=16, out_features=1, bias=True)
)
```

<!-- cell:53 type:code -->
```python
finaloutput, init_output, mid_output = model8.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
plt.title("input dim 1, 1 hidden layers width 16, linear output");
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-46-output-1.png)

<!-- cell:54 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
![Figure](https://rahuldave.com/posts/nnreg/index_files/figure-html/cell-47-output-1.png)
