# core/operational_threshold.py

import numpy as np
from sklearn.linear_model import LinearRegression


class OperationalThreshold:
    def __init__(self):
        self.reg = LinearRegression()
        self.fitted = False
        self.alpha = 0
        self.beta = 0

    def fit(self, O_vals, E_vals):
        O_vals = np.array(O_vals).reshape(-1, 1)
        E_vals = np.array(E_vals).reshape(-1, 1)

        self.reg.fit(O_vals, E_vals)
        self.alpha = self.reg.coef_[0][0]
        self.beta = self.reg.intercept_[0]
        self.fitted = True

    def threshold(self, O_t):
        return self.alpha * O_t + self.beta