# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start = -10, stop = 10, num = 100)
fx = 1/(2*np.pi) / (np.cosh(x/2))
gx = 1/(np.sqrt(2*np.pi))*np.exp(-x**2/2)

plt.plot(x,gx, lw=7, label = "Gaussian Dist")
plt.plot(x,fx, lw=7, label="HSD")
plt.legend()
plt.savefig("GD_HSD.png")
plt.show()

plt.plot(x,gx, lw=7, label = "Gaussian Dist")
plt.plot(x,fx, lw=7, label="HSD")
# plt.legend()
plt.savefig("GD_HSD.png")
plt.show()
