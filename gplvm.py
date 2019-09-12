import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

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

capture_time = y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
time = capture_time.log2() /6

X_prior_mean = torch.zeros(y.size(1), 2)
X_prior_mean[:, 0] = time #もう一行はなんだ？←ここはあえてあけているのだな。

kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
X = Parameter(X_prior_mean.clone())

Xu = stats.resample(X_prior_mean.clone(), 32)
gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01), jitter=1e-5)

# set prior and guide
gplvm.set_prior("X", dist.Normal(X_prior_mean, 0.1).to_event())
gplvm.autoguide("X", dist.Normal)

# note that training is expected to take a minute or so
losses = gp.util.train(gplvm, num_steps=4000)

# let's plot the loss curve after 4000 steps of training
plt.plot(losses)
plt.show()

plt.figure(figsize=(8, 6))
colors = plt.get_cmap("tab10").colors[::-1]
labels = df.index.unique()

X_sampled = gplvm.X_loc.detach().numpy()
for i, label in enumerate(labels):
    X_i = X_sampled[df.index == label]
    plt.scatter(X_i[:, 0], X_i[:, 1], c=colors[i], label=label)

plt.legend()
plt.xlabel("pseudotime", fontsize=14)
plt.ylabel("branching", fontsize=14)
plt.title("GPLVM on Single-Cell qPCR data", fontsize=16)
plt.show()

# ちなみに、主成分分析だとどうなる?
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

y_pre = scale(y, axis=0)
pca_res = PCA(n_components=3)
pca_res.fit(y_pre)


for i, label in enumerate(labels):
    X_i = (pca_res.components_.T)[df.index == label]
    plt.scatter(X_i[:, 0], X_i[:, 1], c=colors[i], label=label)

plt.legend()
plt.xlabel("pseudotime", fontsize=14)
plt.ylabel("branching", fontsize=14)
plt.title("GPLVM on Single-Cell qPCR data", fontsize=16)
plt.show()
# 全然わけられてないorz...