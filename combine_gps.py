import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
from scipy.cluster.vq import kmeans2
from pyro.contrib.gp.util import train
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
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from pyro.contrib import autoname


class Linear(Parameterized):
    def __init__(self, a):
        super(Linear, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.))
        # self.set_prior("a", dist.Normal(0, 10))
        # self.set_prior("a", dist.Cauchy(torch.tensor([0.0]), torch.tensor([10.0])))
        # self.a = pyro.sample("a", dist.Cauchy(torch.tensor([0.0]), torch.tensor([10.0])))
    def forward(self, x):
        # a = pyro.sample("a", self._priors['a'])
        # a = self.get_("a")
        # a = linear._priors['a'].rsample()
        return self.a


class mixGP(gp.parameterized.Parameterized):
    def __init__(self, X1, y, X2, use_cuda=True):
        super(mixGP, self).__init__()
        datasize1, datasize2 = [X1.shape[1], X2.shape[1]]

        linear1 = Linear(torch.tensor(0.))

        layer1_kernel = gp.kernels.RBF(datasize1, variance=torch.tensor(2.), lengthscale=torch.tensor(2.))
        self.layer1 = gp.models.GPRegression(X1, None, layer1_kernel, mean_function=linear1, noise=torch.tensor(1.))
        self.layer1.kernel.set_prior("variance", dist.Exponential(1))
        self.layer1.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
        layer2_kernel = gp.kernels.RBF(datasize2, variance=torch.tensor(2.), lengthscale=torch.tensor(2.))
        self.layer2 = gp.models.GPRegression(X2, None, layer2_kernel, noise=torch.tensor(1.))
        self.layer2.kernel.set_prior("variance", dist.Exponential(1))
        self.layer2.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
        if use_cuda:
            self.layer1.cuda()
            self.layer2.cuda()

    @autoname.name_count
    def model(self, X1, X2, y):
        self.layer1.set_data(X1, None)
        self.layer2.set_data(X2, None)
        f, f_var = self.layer1.model()
        f2, f2_var = self.layer2.model()
        noise = pyro.sample('e1', dist.HalfCauchy(torch.tensor(2.5)))
        with pyro.plate("map", len(X1)):
            pyro.sample("y", dist.Cauchy(f2 + f, f_var + f2_var + noise), obs=y)

    @autoname.name_count
    def guide(self, X1, X2, y):
        self.layer1.set_data(X1, None)
        self.layer2.set_data(X2, None)
        pyro.sample('e1', dist.HalfCauchy(torch.tensor(2.5)))
        self.layer1.guide()
        self.layer2.guide()

    def forward(self, X1, X2):
        pred = []
        for _ in range(100):
            h_loc, h_var = self.layer1(X1)
            h2_loc, h2_var = self.layer2.forward(X2)
            pred.append(h_loc+h2_loc)
        return torch.stack(pred).mode(dim=0)[0]

nsample = 300
nloss = 100
dimx1 = 1
dimx2 = 1

x1_pre = np.random.normal(0, 2, [nsample, dimx1])
x2_pre = np.random.normal(2, 1, [nsample, dimx2])

e1 = np.random.normal(0, 0.5, [nsample])
e2 = np.random.normal(0, 1, [nsample])
w1 = np.random.normal(0, 1, [dimx1, 1])
w2 = np.random.normal(0, 1, [dimx2, 1])
_y1_pre = x1_pre.dot(w1)[:, 0]
_y2_pre = x2_pre.dot(w2)[:, 0]

y1_pre = _y1_pre * np.sin(_y1_pre) + e1 +3
y2_pre = _y2_pre * np.sin(_y2_pre) + y1_pre + e2 -4

x1 = torch.tensor(x1_pre).float()
x2 = torch.tensor(x2_pre).float()
y1 = torch.tensor(y1_pre).float()
y = torch.tensor(y2_pre).float()
y[0:nloss] = torch.tensor(np.tile(np.nan, [nloss]))
pos = ~torch.isnan(y)

self = mixGP(x1[pos], y[pos], x2[pos])
self.cuda()

optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 3000
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(self.model, self.guide, x1[pos], x2[pos], y[pos])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i%500==0:
        print('Now iteration {}: losses: {}'.format(i, loss.item()))
plt.plot(losses)

pred = self.forward(x1[~pos], x2[~pos])
