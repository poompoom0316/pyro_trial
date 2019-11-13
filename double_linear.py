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
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC


nsample = 300
nloss = 50
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

y1_pre = _y1_pre + e1 + 4
y2_pre = _y2_pre + 0.3* y1_pre + e2 - 2

x1 = torch.tensor(x1_pre).float()
x2 = torch.tensor(x2_pre).float()
x = torch.tensor(x_pre).float()
y1 = torch.tensor(y1_pre).float()
y2 = torch.tensor(y2_pre).float()
y1[0:nloss] = torch.tensor(np.tile(np.nan, [nloss]))
y2[20:(nloss + 20)] = torch.tensor(np.tile(np.nan, [nloss]))
y = torch.stack([y1, y2], 1)

pos = ~torch.isnan(y1)
pos2 = ~torch.isnan(y2)


class Double_Linear_Model(nn.Module):
    def __init__(self):
        super(Double_Linear_Model, self).__init__()

    def model(self, x1, x2, y):
        a = pyro.sample("a", dist.Cauchy(torch.tensor([0., 0.]), torch.tensor(10.)))
        b_1 = pyro.sample("b1", dist.Cauchy(torch.zeros([x1.shape[1], 1]), torch.tensor(2.5)))
        b_2 = pyro.sample("b2", dist.Cauchy(torch.zeros([x2.shape[1], 1]), torch.tensor(2.5)))
        b_ar = pyro.sample("bt", dist.Cauchy(torch.tensor([0.]), torch.tensor(2.5)))
        sigma = pyro.sample("sigma1", dist.HalfCauchy(25))
        sigma2 = pyro.sample("sigma2", dist.HalfCauchy(25))
        mean1 = (a[0] + x1.mm(b_1)).squeeze(-1)
        mean2 = (a[1] + x1.mm(b_2)).squeeze(-1) + b_ar*mean1

        y1_impute = pyro.sample('y1_impute', dist.Normal(mean1, sigma))
        y2_impute = pyro.sample('y2_impute', dist.Normal(mean2, sigma))
        isnan1 = torch.isnan(y[:, 0])
        isnan2 = torch.isnan(y[:, 1])
        y1 = y[:, 0]
        y2 = y[:, 1]
        y1[isnan1] = y1_impute[isnan1]  # update x
        y2[isnan2] = y2_impute[isnan2]  # update x

        pyro.sample("obs1", dist.Normal(mean1, sigma), obs=y1)
        pyro.sample("obs2", dist.Normal(mean2, sigma2), obs=y2)

        # return [pyro.sample("obs1", dist.Normal(mean1, sigma), obs=y[:, 0]),
        #         pyro.sample("obs2", dist.Normal(mean2, sigma2), obs=y[:, 1])]


dlr = Double_Linear_Model()

nuts_kernel = NUTS(dlr.model, jit_compile=True)
mcmc2 = MCMC(nuts_kernel,
            num_samples=100,
            warmup_steps=20,
            num_chains=1)
mcmc2.run(x1, x2, y)
mcmc2.summary(prob=0.95)
