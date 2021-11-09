import gpflow
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)  # for reproducibility of this notebook

plt.style.use("ggplot")


def plotkernelsample(k, ax, xmin=-3, xmax=3, title=None):
    xx = np.linspace(xmin, xmax, 100)[:, None]
    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), k(xx), 3).T)
    ax.set_title(title)
    plt.show()

np.random.seed(1)

base_k1 = gpflow.kernels.Matern32(lengthscales=0.2)
base_k2 = gpflow.kernels.Matern32(lengthscales=2.0)
k = gpflow.kernels.ChangePoints([base_k1, base_k2], [0.0], steepness=5.0)

f, ax = plt.subplots(1, 1, figsize=(10, 3))
plotkernelsample(k, ax)