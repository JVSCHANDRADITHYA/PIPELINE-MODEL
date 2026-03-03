# core/pca_stream.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GroupPCA:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        self.fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.fitted = True

    def transform(self, x_row):
        x_scaled = self.scaler.transform([x_row])
        return self.pca.transform(x_scaled)[0][0]