"""
VIS_DATA
Visualize different types of data

Stefan Wong 2017
"""

import numpy as np
import matplotlib.pyplot as plt


def vis_synth_data(X, y, ax=None, title_text=None):

    if ax is None:
        return

    ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
    ax.set_title(str(title_text))
