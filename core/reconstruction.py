# core/reconstruction.py

import numpy as np


def reconstruction_error(window):
    """
    window shape: (L, 3)
    """
    mean = np.mean(window, axis=0)
    reconstructed = np.tile(mean, (window.shape[0], 1))
    return np.mean((window - reconstructed) ** 2)