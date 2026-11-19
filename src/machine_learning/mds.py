import numpy as np
import matplotlib.pyplot as plt

from .pca import PCA

class MDS:
    def __init__(self, distance: np.ndarray):
        self.distance = distance
        self.embedding = None

    def fit(self, n_components: int = 2):
        n_samples = self.distance.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        Q = -0.5 * H @ (self.distance ** 2) @ H
        pca = PCA(Q)
        self.embedding = pca.fit_transform(n_components)
        