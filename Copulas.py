from copulas.visualization import scatter_2d
import pandas as pd
import matplotlib.pylab as plt
from pycopula.copula import ArchimedeanCopula
from pycopula.visualization import pdf_2d, cdf_2d
import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm
from numpy.linalg import inv, cholesky
from scipy.stats import multivariate_normal, invgamma, t as student
import math

def simulate(copula, n):
    """
    Generates random variables with selected copula's structure.
    Parameters
    ----------
    copula : Copula
        The Copula to sample.
    n : integer
        The size of the sample.
    """
    d = copula.dimension()

    X = []
    if type(copula).__name__ == "Copula" and copula.name == "indep":
        for i in range(n):
            X.append([np.random.uniform() for j in range(d)])
    elif type(copula).__name__ == "Copula" and copula.name == "frechet_up":
        for i in range(n):
            Xi = np.random.uniform(size=d)
            Xi = np.full_like(Xi, Xi.min())
            X.append(Xi)
    elif type(copula).__name__ == "Copula" and copula.name == "frechet_down":
        if d != 2:
            raise ValueError("Fréchet lower bound is not a copula for dimensions other than 2")
        for i in range(n):
            Xi = np.random.uniform(size=2)
            Xi[1] = 1 - Xi[0]
            X.append(Xi)
    elif type(copula).__name__ == "GaussianCopula":
        # We get correlation matrix from covariance matrix
        Sigma = copula.get_corr()
        D = sqrtm(np.diag(np.diag(Sigma)))
        Dinv = inv(D)
        P = np.dot(np.dot(Dinv, Sigma), Dinv)
        A = cholesky(P)

        for i in range(n):
            Z = np.random.normal(size=d)
            V = np.dot(A, Z)
            U = stats.norm.cdf(V)
            X.append(U)
    elif type(copula).__name__ == "ArchimedeanCopula":
        U = np.random.rand(n, d)

        # LaplaceâStieltjes invert transform
        LSinv = {'clayton': lambda theta: np.random.gamma(shape=1. / theta),
                 'gumbel': lambda theta: stats.levy_stable.rvs(1. / theta, 1., 0,
                                                               math.cos(math.pi / (2 * theta)) ** theta),
                 'frank': lambda theta: stats.logser.rvs(1. - math.exp(-theta)),
                 'amh': lambda theta: stats.geom.rvs(theta)}

        for i in range(n):
            V = LSinv[copula.getFamily()](copula.get_parameter())
            X_i = [copula.inverse_generator(-np.log(u) / V) for u in U[i, :]]
            X.append(X_i)
    elif type(copula).__name__ == "StudentCopula":
        nu = copula.get_df()
        Sigma = copula.get_corr()

        for i in range(n):
            Z = multivariate_normal.rvs(size=1, cov=Sigma)
            W = invgamma.rvs(nu / 2., size=1)
            U = np.sqrt(W) * Z
            X_i = [student.cdf(u, nu) for u in U]
            X.append(X_i)

    return X

path = "/Users/aslanramazanov/Desktop/data_excel.xlsx"
data = pd.ExcelFile(path).parse()
dataset = pd.DataFrame(data)

#columns_0 = ['ln_XOM_ret','ln_JPM_ret']
#returns = pd.DataFrame(dataset, columns=columns_0)

columns = ['ln_AAPL_ret_norm','ln_XOM_ret_norm']
returns_norm = pd.DataFrame(dataset, columns=columns)
returns_np = returns_norm.to_numpy()

#scatter_2d(returns_norm)
#plt.show()

clayton = ArchimedeanCopula(family="clayton", dim=2)
clayton.fit(returns_np, method="cmle", df_fixed=False)

synthetic_data = pd.DataFrame(simulate(clayton, 2014))

scatter_2d(synthetic_data)
plt.show()

print(clayton)

# Plotting
u, v, C = cdf_2d(clayton)
u, v, c = pdf_2d(clayton)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Clayton copula CDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap='Blues')
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)

ax = fig.add_subplot(122, title="Clayton copula PDF")
ax.contour(X, Y, c, levels = np.arange(0,5,0.15))

plt.show()



gumbel = ArchimedeanCopula(family="gumbel", dim=2)
gumbel.fit(returns_np, method="cmle", df_fixed=False)

synthetic_data_1 = pd.DataFrame(simulate(clayton, 2014))

scatter_2d(synthetic_data_1)
plt.show()

print(gumbel)

u, v, C = cdf_2d(gumbel)
u, v, c = pdf_2d(gumbel)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Gumbel copula CDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap='Blues')
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)

ax = fig.add_subplot(122, title="Gumbel copula PDF")
ax.contour(X, Y, c, levels = np.arange(0,5,0.15))

plt.show()













