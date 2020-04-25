import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# setting up model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, training_iter=100):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()

    train_x = train_x.cuda()
    train_y = train_y.cuda()

    # training the model
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # adamで最適化
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
    ], lr=0.05)

    # defining loss og GPs (marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i+1}/{training_iter} - Loss: .3{loss.item()} lengthscale: "
              f"{model.covar_module.base_kernel.lengthscale.item()}, noise:{model.likelihood.noise.item()}")
        optimizer.step()

    return model, likelihood


def test(test_x, model, likelihood):
    model.eval()
    likelihood.eval()
    test_x = test_x.cuda()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    return observed_pred, mean, lower, upper


def make_plot(mean, lower, upper, train_x, train_y, test_x):
    mean2 = mean.cpu()
    lower2 = lower.cpu()
    upper2 = upper.cpu()

    train_x2 = train_x.cpu()
    train_y2 = train_y.cpu()
    test_x2 = test_x.cpu()

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x2.numpy(), train_y2.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x2.numpy(), mean2.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x2.numpy(), lower2.numpy(), upper2.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])


def main():
    train_x = torch.linspace(0, 1, 10000)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.5

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()

    train_x = train_x.cuda()
    train_y = train_y.cuda()

    # training the model
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # adamで最適化
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
    ], lr=0.05)

    # defining loss og GPs (marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i+1}/{training_iter} - Loss: .3{loss.item()} lengthscale: "
              f"{model.covar_module.base_kernel.lengthscale.item()}, noise:{model.likelihood.noise.item()}")
        optimizer.step()

    test_x = torch.linspace(0, 1, 51).cuda()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    make_plot(mean, lower, upper, train_x, train_y, test_x)