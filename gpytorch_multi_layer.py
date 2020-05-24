import math
import torch
import gpytorch

import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats
import pyro.infer.mcmc as mcmc
import os
import tqdm
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt


def prepare_data(ndata=100, dim_p=2, nlayer=3):
    np.random.seed(777)

    X_pre = np.random.random([ndata, dim_p, nlayer])
    y0 = np.random.random([ndata, 1])
    eps = np.random.normal(size=ndata).reshape([ndata, 1])
    Y_pre = np.concatenate([y0, np.zeros([ndata, nlayer])], axis=1)
    Y = np.concatenate([y0 + eps, np.zeros([ndata, nlayer])], axis=1)

    ws = np.random.random([dim_p + 1, nlayer])
    for i in range(nlayer):
        yi = Y_pre[:, i:(i + 1)]
        epsi = np.random.normal(size=ndata)
        arrayi = np.c_[X_pre[:, :, i], yi]
        mat_xw = np.array([arrayi[:, k] * ws[k, i:(i + 1)] for k in range(arrayi.shape[1])])
        xsinx = mat_xw * np.sin(mat_xw)
        yi_previous = xsinx.prod(0)
        Y_pre[:, i + 1] = yi_previous

        Y[:, i + 1] = np.exp(yi_previous + epsi)

    return X_pre, Y_pre, Y


class PVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, ndim=1, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.shape[0])
        )

        # Standard initializtation
        # super().__init__(variational_strategy)
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = function_samples.exp()

            # Sample from observed distribution
            # return pyro.sample(
            #     self.name_prefix + ".y",
            #     pyro.distributions.Exponential(scale_samples.reciprocal()),  # rate = 1 / scale
            #     obs=y
            # )
            return pyro.sample(
                self.name_prefix + ".y",
                function_dist,  # rate = 1 / scale
                obs=y
            )


def train2(x, y, prev_model_list, x_pre_list, num_particles=256, num_iter=200):
    if prev_model_list is not None:
        for i, (prev_model, x_pre) in enumerate(zip(prev_model_list, x_pre_list)):
            prev_model.eval()
            if i == 0:
                with torch.no_grad():
                    output = prev_model(x_pre)
            else:
                x_pre2 = torch.cat((x_pre, mean1.reshape([mean1.shape[0], 1])), dim=1)
                with torch.no_grad():
                    output = prev_model(x_pre2)
            mean1 = output.mean
            x_post = torch.cat((x, mean1.reshape([mean1.shape[0], 1])), dim=1)
    else:
        x_post = x

    model = PVGPRegressionModel(inducing_points=x_post[0:128], ndim=x_post.shape[1])

    optimizer = pyro.optim.Adam({"lr": 0.1})
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()
    iterator = range(num_iter)
    loss_list = []
    for i in iterator:
        model.zero_grad()
        loss = svi.step(x_post, y)
        loss_list.append(loss)
    return model, loss_list


def predict2(x, model, prev_model_list, x_pre_list):
    if prev_model_list is not None:
        for i, (prev_model, x_pre) in enumerate(zip(prev_model_list, x_pre_list)):
            prev_model.eval()
            if i == 0:
                with torch.no_grad():
                    output = prev_model(x_pre)
            else:
                x_pre2 = torch.cat((x_pre, mean1.reshape([mean1.shape[0], 1])), dim=1)
                with torch.no_grad():
                    output = prev_model(x_pre2)
            mean1 = output.mean
        x_post = torch.cat((x, mean1.reshape([mean1.shape[0], 1])), dim=1)
    else:
        x_post = x

    with torch.no_grad():
        output = model(x_post)

    return output


def main_pre():
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
    y2_pre = _y2_pre * np.sin(_y2_pre) + _y1_pre + e2 - 2

    x1 = torch.tensor(x1_pre).float()
    x2 = torch.tensor(x2_pre).float()
    x = torch.tensor(x_pre).float()
    y1 = torch.tensor(y1_pre).float()
    y2 = torch.tensor(y2_pre).float()
    y1[0:nloss] = torch.tensor(np.tile(np.nan, [nloss]))
    y2[20:(nloss + 20)] = torch.tensor(np.tile(np.nan, [nloss]))

    pos = ~torch.isnan(y1)
    pos2 = ~torch.isnan(y2)

    model1, loss1 = train2(x=x1[pos], y=y1[pos], prev_model_list=None, x_pre_list=None, num_iter=500)
    model2, loss2 = train2(x=x2[pos2], y=y2[pos2], prev_model_list=[model1], x_pre_list=[x1[pos2]], num_iter=500)

    pred_output = predict2(x=x2[~pos2], model=model2, prev_model_list=[model1], x_pre_list=[x1[~pos2]])
    pred_output2 = predict2(x=x2[pos2], model=model2, prev_model_list=[model1], x_pre_list=[x1[pos2]])
    pred_output3 = predict2(x=x1[pos], model=model1, prev_model_list=None, x_pre_list=None)
