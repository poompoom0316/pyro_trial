import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

from pyro.distributions import Normal, Uniform, Delta, HalfCauchy, Cauchy

datasize = 100

def yfunc(x, w):
    y = (w[:, 0]+w[:, 1]*x+w[:, 2]*x**2)/(1+w[:, 3]*x)
    return y.reshape(x.shape[0], 1)


def data():
    ws_loc = [8.5, -2.9, -0.2, -0.1]
    ws_std = [0.6, 0.3, 0.05, 0.05]
    ws = np.concatenate(
        [np.random.normal(loci, scalei, size=datasize).reshape([datasize, 1]) for loci, scalei in zip(ws_loc, ws_std)],
        axis=1)

    x1 = np.random.random(datasize) *0.25 +0.1
    x2 = x1 + np.random.random(datasize) *0.25
    x3 = x2 + np.random.random(datasize) *0.25
    x4 = x3 + np.random.random(datasize) *0.25
    x_pre = (np.concatenate([x1.reshape([datasize, 1]), x2.reshape([datasize, 1]), x3.reshape([datasize, 1]), x4.reshape([datasize, 1])], axis=1)*1600)**0.5

    std_error = 50
    errors = np.random.normal(0, std_error, 4*x1.shape[0]).reshape([x1.shape[0], 4])
    ys_pre = np.concatenate([yfunc(x_pre[:, i], ws) for i in range(4)], axis=1)

    y_pre = ys_pre+errors

    x_data = torch.tensor(x_pre, dtype=torch.float32)
    y_data = torch.tensor(y_pre, dtype=torch.float32)

    return x_data, y_data


# Bayesian regression
def model(x_data, y_data):
    scalew = pyro.sample("sigmaw", HalfCauchy(1))
    w_prior = pyro.sample('ws', Normal(torch.zeros(3), scalew))
    b_prior = pyro.sample("b", Cauchy(torch.tensor([[0.]]), torch.tensor([[10.]])))

    scale1 = pyro.sample("sigma_w1", HalfCauchy(10))
    scale2 = pyro.sample("sigma_w2", HalfCauchy(10))
    scale3 = pyro.sample("sigma_w3", HalfCauchy(10))
    scale4 = pyro.sample("sigma_b", HalfCauchy(10))
    scale = pyro.sample("sigma", HalfCauchy(1))

    bvec = torch.tensor([pyro.sample("b_{}".format(i), Normal(b_prior, scale4)) for i in range(datasize)],
                         requires_grad=True)
    wvec1 = torch.tensor([pyro.sample("w1_{}".format(i), Normal(w_prior[0], scale1)) for i in range(datasize)],
                         requires_grad=True)
    wvec2 = torch.tensor([pyro.sample("w2_{}".format(i), Normal(w_prior[1], scale2)) for i in range(datasize)],
                         requires_grad=True)
    wvec3 = torch.tensor([pyro.sample("w3_{}".format(i), Normal(w_prior[2], scale3)) for i in range(datasize)],
                         requires_grad=True)

    prediction_mean = torch.cat([((bvec+wvec1*x_data[:, i])/(1+wvec3*x_data[:, i])).reshape([datasize, 1])
                                 for i in range(4)], axis=1)
    return pyro.sample("obs", Normal(prediction_mean.flatten(), scale), obs=y_data.flatten())


def conditioned_model(model, x_data, y_data):
    return poutine.condition(model, data={"obs": y_data.flatten()})(x_data.flatten())


def main(args):
    x_data, y_data = data()

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel,
                num_samples=1000,
                num_chains=2)
    mcmc.run(x_data, y_data)
    mcmc.summary(prob=0.5)