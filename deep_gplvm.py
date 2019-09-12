import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats
import pyro.infer.mcmc as mcmc

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('0.4.0')
pyro.enable_validation(True)       # can help with debugging
pyro.set_rng_seed(1)

# license: Copyright (c) 2014, the Open Data Science Initiative
# license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"

df = pd.read_csv(URL, index_col=0)
print("Data shape: {}\n{}\n".format(df.shape, "-" * 21))
print("Data labels: {}\n{}\n".format(df.index.unique().tolist(), "-" * 86))
print("Show a small subset of the data:")
df.head()

data = torch.tensor(df.values, dtype=torch.get_default_dtype())
y = data.t()
N = y.shape[1]

capture_time = y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
time = capture_time.log2() /6

X_prior_mean = torch.zeros(y.size(1), 15)
X_prior_mean[:, 0] = time #もう一行はなんだ？←ここはあえてあけているのだな。

kernel = gp.kernels.RBF(input_dim=15, lengthscale=torch.ones(15))
kernel2 = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
# we setup the mean of our prior over X
X = Parameter(X_prior_mean.clone())
Xu = stats.resample(X_prior_mean.clone(), 100)

# we setup the mean of our prior over Z
Z_prior_mean = torch.zeros(y.size(1), 2)  # shape: 437 x 2
Z_prior_mean[:, 0] = time
h = Parameter(Z_prior_mean.clone())

# first layer
gpr1 = gp.models.GPRegression(X, h.t(), kernel, noise=torch.tensor(10**(-3)),
                              mean_function=lambda x: x) # mean_function???

gpr1.kernel.set_prior("variance", dist.Exponential(10))
gpr1.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
gpr1.set_prior("X", dist.Normal(X_prior_mean, 0.1).to_event())
gpr1.autoguide("X", dist.Normal)

# second layer
gpr2 = gp.models.GPRegression(h, y, kernel2, noise=torch.tensor(1e-3))
gpr2.kernel.set_prior("variance", dist.Exponential(1))
gpr2.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))


def model():
    h_loc, h_var = gpr1.model()
    gpr2.X = pyro.sample("h", dist.Normal(h_loc, h_var.sqrt()))
    gpr2.model()


hmc_kernel = mcmc.NUTS(model)
posterior = mcmc.MCMC(hmc_kernel, num_samples=100).run()


