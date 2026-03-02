<!-- cell:1 type:markdown -->
# How sigmoids combine


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
[Figure]

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
[Figure]

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
dataset.data_tensor.shape, dataset.target_tensor.shape
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
            localaccum.append(loss.data[0])
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
  (fc_initial): Linear(in_features=1, out_features=2)
  (fc_mid): ModuleList(
    (0): Linear(in_features=2, out_features=2)
  )
  (fc_final): Linear(in_features=2, out_features=1)
)
```

<!-- cell:13 type:code -->
```python
accum = run_model(model, 2000)
```

<!-- cell:14 type:code -->
```python
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:15 type:code -->
```python
plt.plot([a[0]+a[1] for a in accum]);
plt.plot([a[0]-a[1] for a in accum]);
plt.xlim(0, 1000)
```
Output:
```
(0, 1000)
```
[Figure]

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
[<matplotlib.lines.Line2D at 0x122320c88>]
```
[Figure]

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
[Figure]

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
  (fc_initial): Linear(in_features=1, out_features=4)
  (fc_mid): ModuleList(
    (0): Linear(in_features=4, out_features=4)
  )
  (fc_final): Linear(in_features=4, out_features=1)
)
```

<!-- cell:22 type:code -->
```python
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:23 type:code -->
```python
finaloutput, init_output, mid_output = model2.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
```
Output:
```
[<matplotlib.lines.Line2D at 0x12304ae80>]
```
[Figure]

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
[Figure]

<!-- cell:26 type:markdown -->
### input dim 1, 2 hidden layers width 8, linear output

<!-- cell:27 type:code -->
```python
model3 = MLRegP(1, 8, nonlinearity=fn.sigmoid, additional_hidden_wide=1)
accum = run_model(model3, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:28 type:code -->
```python
print(model3)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=8)
  (fc_mid): ModuleList(
    (0): Linear(in_features=8, out_features=8)
  )
  (fc_final): Linear(in_features=8, out_features=1)
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
[<matplotlib.lines.Line2D at 0x1220cc630>]
```
[Figure]

<!-- cell:30 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
[Figure]

<!-- cell:31 type:markdown -->
### input dim 1, 3 hidden layers width 4, linear output

<!-- cell:32 type:code -->
```python
model4 = MLRegP(1, 4, nonlinearity=fn.sigmoid, additional_hidden_wide=2)
accum = run_model(model4, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:33 type:code -->
```python
print(model4)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=4)
  (fc_mid): ModuleList(
    (0): Linear(in_features=4, out_features=4)
    (1): Linear(in_features=4, out_features=4)
  )
  (fc_final): Linear(in_features=4, out_features=1)
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
[<matplotlib.lines.Line2D at 0x12249c780>]
```
[Figure]

<!-- cell:35 type:markdown -->
### input dim 1, 3 hidden layers width 2, linear output

<!-- cell:36 type:code -->
```python
model5 = MLRegP(1, 2, nonlinearity=fn.sigmoid, additional_hidden_wide=2)
accum = run_model(model5, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:37 type:code -->
```python
print(model5)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=2)
  (fc_mid): ModuleList(
    (0): Linear(in_features=2, out_features=2)
    (1): Linear(in_features=2, out_features=2)
  )
  (fc_final): Linear(in_features=2, out_features=1)
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
[<matplotlib.lines.Line2D at 0x12349d7b8>]
```
[Figure]

<!-- cell:39 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
[Figure]

<!-- cell:40 type:markdown -->
### input dim 1, 1 hidden layers width 2, linear output

<!-- cell:41 type:code -->
```python
model6 = MLRegP(1, 2, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model6, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:42 type:code -->
```python
print(model6)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=2)
  (fc_mid): ModuleList(
  )
  (fc_final): Linear(in_features=2, out_features=1)
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
[<matplotlib.lines.Line2D at 0x123365cc0>]
```
[Figure]

<!-- cell:44 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
[Figure]

<!-- cell:45 type:markdown -->
### input dim 1, 1 hidden layers width 1, linear output

<!-- cell:46 type:code -->
```python
model7 = MLRegP(1, 1, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model7, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:47 type:code -->
```python
print(model7)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=1)
  (fc_mid): ModuleList(
  )
  (fc_final): Linear(in_features=1, out_features=1)
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
[<matplotlib.lines.Line2D at 0x121f65b70>]
```
[Figure]

<!-- cell:49 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
[Figure]

<!-- cell:50 type:markdown -->
### input dim 1, 1 hidden layers width 16, linear output

<!-- cell:51 type:code -->
```python
model8 = MLRegP(1, 16, nonlinearity=fn.sigmoid, additional_hidden_wide=0)
accum = run_model(model8, 4000)
plt.plot([a[0] for a in accum]);
```
[Figure]

<!-- cell:52 type:code -->
```python
print(model8)
```
Output:
```
MLRegP(
  (fc_initial): Linear(in_features=1, out_features=16)
  (fc_mid): ModuleList(
  )
  (fc_final): Linear(in_features=16, out_features=1)
)
```

<!-- cell:53 type:code -->
```python
finaloutput, init_output, mid_output = model8.forward(xdata.view(-1,1))
plt.plot(xgrid, ygrid, '.')
plt.plot(xgrid, finaloutput.data.numpy(), lw=3, color="r")
plt.title("input dim 1, 1 hidden layers width 16, linear output");
```
[Figure]

<!-- cell:54 type:code -->
```python
io = mid_output.data.numpy()
plt.plot(xgrid, ygrid, '.', alpha=0.2)
for j in range(io.shape[1]):
    plt.plot(xgrid, io[:, j], lw=2)
```
[Figure]
