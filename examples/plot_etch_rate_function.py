# %%
"""
Plot the calibrated etch rate model V(D).

Loads the globally calibrated etch rate model and plots the V(D) curve,
including the 1-sigma uncertainty band when uncertainties are available.
"""

import matplotlib.pyplot as plt
from tracketch import load_etchrate_model

etch_model = load_etchrate_model("Doerschel_etching")

fig, ax = etch_model.plot()
ax.set_title("Calibrated etch rate model")
plt.show()
