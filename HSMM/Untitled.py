# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def multi_hyperbolic_secant(x:np.ndarray, s:float):
    return 1/np.cosh(np.sqrt(s*np.abs(x[:,0])*np.abs(x[:,1]))/2) * np.sqrt(s)/(2*np.pi)**2


# +
m = 2 #dimension
s = 2

N = 100
x1 = np.linspace(-10, 10, N)
x2 = np.linspace(-10, 10, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multi_hyperbolic_secant(x=X, s=s)
Y_plot = Y_plot.reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Y_plot, cmap='bwr', linewidth=0)
fig.colorbar(surf)
ax.set_title("Surface Plot")
fig.show()

# -

Y_plot.sum()


