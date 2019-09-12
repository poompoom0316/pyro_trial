import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
from scipy.cluster.vq import kmeans2
import copy
import torch
import torch.nn as nn
from torch.distributions.transforms import AffineTransform
from torch.distributions import half_cauchy
from torchvision import transforms
from torch.nn import Parameter
import pyro
import pyro.contrib.gp as gp
from pyro.contrib.gp.parameterized import Parameterized
from pyro.infer import SVI, TraceGraph_ELBO, EmpiricalMarginal
import pyro.distributions as dist
import pyro.infer as infer
import pyro.infer.mcmc as mcmc

nsample = 300
nloss = 100
dimx1 = 1
dimx2 = 1

x1_pre = np.random.normal(0, 2, [nsample, dimx1])
x2_pre = np.random.normal(2, 1, [nsample, dimx2])
x_pre = np.c_[x1_pre, x2_pre]

e1 = np.random.normal(0, 0.5, [nsample])
e2 = np.random.normal(0, 0.5, [nsample])
w1 = np.random.normal(0, 1, [dimx1, 1])
w2 = np.random.normal(0, 1, [dimx2, 1])
_y1_pre = x1_pre.dot(w1)[:, 0]
_y2_pre = x2_pre.dot(w2)[:, 0]

y1_pre = _y1_pre * np.sin(_y1_pre) + e1 + 4
y2_pre = _y2_pre * np.sin(_y2_pre) + y1_pre + e2 - 2

x1 = torch.tensor(x1_pre).float()
x2 = torch.tensor(x2_pre).float()
x = torch.tensor(x_pre).float()
y1 = torch.tensor(y1_pre).float()
y2 = torch.tensor(y2_pre).float()
y1[0:nloss] = torch.tensor(np.tile(np.nan, [nloss]))
y2[20:(nloss + 20)] = torch.tensor(np.tile(np.nan, [nloss]))

pos = ~torch.isnan(y1)
pos2 = ~torch.isnan(y2)

# plt.plot(x1_pre[:,0], y1_pre, 'o')


# first step
gpr1 = gp.models.GPRegression(x1[pos], y1[pos], gp.kernels.RBF(1), noise=torch.tensor(1.))
gpr1.kernel.set_prior("variance", dist.Exponential(1))
gpr1.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
# gpr1.cuda()

optimizer = torch.optim.Adam(gpr1.parameters(), lr=0.005)

loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 5000
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr1.model, gpr1.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
# plt.plot(losses)

pred, _ = gpr1.forward(x1[~pos])
plt.plot(y1_pre[0:nloss], pred.detach().numpy(), 'o')
print("RMSE: {}".format(np.mean((y1_pre[0:nloss] - pred.detach().numpy()) ** 2) ** 0.5))
print(1 - np.mean((y1_pre[0:nloss] - pred.detach().numpy()) ** 2) / y1_pre.var())
# step2
gpr2 = gp.models.GPRegression(x2, torch.zeros(nsample), gp.kernels.RBF(1), noise=torch.tensor(1.))
gpr2.kernel.set_prior("variance", dist.Exponential(1))
gpr2.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))


####plan 1: classにすべてを書く####
class DoubleGP(gp.parameterized.Parameterized):
    def __init__(self, X, y, gpr1, dimx1, dimx2):
        super(DoubleGP, self).__init__()
        x1 = x[:, 0:dimx1]
        x2 = x[:, dimx1:(dimx1 + dimx2)]
        self.dimx1 = dimx1
        self.dimx2 = dimx2

        gpr2 = gp.models.GPRegression(x2, None, gp.kernels.RBF(1), noise=torch.tensor(1.))
        gpr2.kernel.set_prior("variance", dist.Exponential(1))
        gpr2.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))

        self.layer1 = gpr1
        self.layer2 = gpr2

    def model(self, x, y):
        x1 = x[:, 0:self.dimx1]
        x2 = x[:, self.dimx1:(self.dimx1 + self.dimx2)]

        y1_loc, y1_var = gpr1.forward(x1)
        y1hat = pyro.sample("y1hat", dist.Normal(y1_loc, y1_var.sqrt()))

        # self.layer2.set_data(x2, None)
        # y_loc, y_var = self.layer2.model()
        self.layer2.set_data(x2, y - y1hat)  # これでよいのか？
        self.layer2.model()

        # pyro.sample("y", dist.Cauchy(y1hat + y_loc, y_var), obs=y)

    def guide(self, x, y):
        x1 = x[:, 0:dimx1]
        x2 = x[:, dimx1:(dimx1 + dimx2)]
        self.layer2.guide()

    def forward(self, X_new):
        pred = []
        x1 = X_new[:, 0:self.dimx1]
        x2 = X_new[:, self.dimx1:(self.dimx1 + self.dimx2)]

        for _ in range(1000):
            h_loc, h_var = self.layer1(x1)
            f_loc, f_var = self.layer2(x2)
            y2 = pyro.sample("y", dist.Normal(h_loc + f_loc, f_var.sqrt()))
            pred.append(y2)
        return torch.stack(pred).mode(dim=0)

        # h_loc, h_var = self.layer1(x1)
        # f_loc, f_var = self.layer2(x2)
        # y2 = pyro.sample("y", dist.Normal(h_loc + f_loc, f_var.sqrt()))
        # return y2


gpr12 = copy.copy(gpr1)
dgp = DoubleGP(x[pos2], y2[pos2], gpr12, dimx1, dimx2)

optimizer2 = torch.optim.Adam(dgp.layer2.parameters(), lr=0.005)

losses2 = []
### gradientで最適化する？
for i in range(5000):
    optimizer2.zero_grad()
    loss = loss_fn(dgp.model, dgp.guide, x[pos2], y2[pos2])
    loss.backward()
    optimizer2.step()
    losses2.append(loss.item())
plt.plot(losses2)
# f1, indices = dgp.forward(x)
f1, _ = gpr12.forward(x1)
f2, _ = dgp.layer2.forward(x2)
f, _ = dgp.forward(x)
plt.plot(y2_pre[~pos2], (f1.detach().numpy() + f2.detach().numpy())[~pos2], 'o')
plt.plot(y2_pre[~pos2], f.detach().numpy()[~pos2], 'o')

print("RMSE: {}".format(np.mean((y2_pre - f1.detach().numpy() -f2.detach().numpy())[~pos2] ** 2) ** 0.5))
print("RMSE: {}".format(np.mean((y2_pre - f.detach().numpy())[~pos2] ** 2) ** 0.5))
