import numpy as np
import matplotlib.pyplot as plt



def draw_fitVals_1Dim(fitVals, labels, xlabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    low_range, high_range = np.min([np.min(arr) for arr in fitVals]), np.max([np.max(arr) for arr in fitVals])

    for fitval, lb in zip(fitVals, labels):
        ax.hist(fitval, bins=50, range=(low_range, high_range), histtype='step', label=lb)
    ax.legend()
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    return fig, ax



def draw_fitVals_2Dim(fitVals1, fitVals2, labels, xlabel, ylabel):
    fig, ax = plt.subplots( figsize=(7, 5) )
    for v1, v2, lb in zip(fitVals1, fitVals2, labels):
        ax.scatter(v1, v2, s=10, alpha=0.3, label=lb)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    return fig, ax


