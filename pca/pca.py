import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from linalg.linalg import SVD

class PCA():
    def __init__(self, X: np.ndarray):
        """
        Initializes a PCA object.

        Attributes:
            X: (np.ndarray) Data matrix of shape (n_samples, n_features).
        """
        self.X = X
        self.n, self.p = X.shape
        self._fitted = False

    def fit(self):
        """
        Fits PCA: centers the data, computes the covariance matrix,
        and performs SVD via the parent class to extract components.

        Sets:
            self.mean_: (np.ndarray) Per-feature means, shape (p,).
            self.components_: (np.ndarray) Principal axes, shape (p, p),
                              columns sorted by descending variance.
            self.explained_variance_: (np.ndarray) Variance along each PC.
            self.explained_variance_ratio_: (np.ndarray) Fraction of total variance.
        """
        self.mean_ = self.X.mean(axis=0)
        self.std_ = self.X.std(axis=0, ddof=1)
        X_centered = (self.X - self.mean_) / self.std_
        svd = SVD(X_centered)
        U, S, V = svd.decompose() 

        idx = np.argsort(S**2)[::-1]
        self.components_ = V[:, idx]           # shape (p, p)
        self.explained_variance_ = (S[idx])**2
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )
        self._fitted = True
        return self

    def transform(self, X: np.ndarray = None, n_components: int = None) -> np.ndarray:
        """
        Projects data onto the principal components.

        Input:
            X: (np.ndarray) Data to transform, shape (n_samples, n_features).
                            Uses training data if None.
            n_components: (int) Number of components to keep. Keeps all if None.

        Returns:
            T: (np.ndarray) Projected data, shape (n_samples, n_components).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if X is None:
            X = self.X
        k = n_components or self.p
        return ((X - self.mean_) / self.std_) @ self.components_[:, :k]

    def fit_transform(self, n_components: int = None) -> np.ndarray:
        """Fit and project the training data in one call."""
        return self.fit().transform(n_components=n_components)

    def summary(self):
        """Prints a variance summary table."""
        if not self._fitted:
            raise RuntimeError("Call fit() before summary().")
        cumvar = np.cumsum(self.explained_variance_ratio_)
        print(f"{'PC':<5} {'Eigenvalue':>12} {'Var %':>8} {'Cum %':>8}")
        print("-" * 36)
        for i, (ev, vr, cv) in enumerate(
            zip(self.explained_variance_, self.explained_variance_ratio_, cumvar), 1
        ):
            print(f"PC{i:<3} {ev:>12.4f} {vr*100:>7.2f}% {cv*100:>7.2f}%")

    def biplot(self, X: np.ndarray = None, scale_factor: int = 1, labels: list = None):
        if not self._fitted:
            raise RuntimeError("Call fit() before biplot().")
        if X is None:
            X = self.X
        projection = self.transform(X, 2)
        variables = self.components_[:, :2] * np.sqrt(self.explained_variance_[:2] / (self.n - 1))
        plt.figure('Biplot')
        plt.plot(projection[:,0], projection[:,1],'.b')        
        for i in range(self.p):
            plt.arrow(0, 0,
                    variables[i, 0]*scale_factor,
                    variables[i, 1]*scale_factor,
                    color='r', head_width=0.05)
            if labels is not None:
                plt.text(variables[i, 0]*scale_factor*1.1, variables[i, 1]*scale_factor*1.1,
                     labels[i], color='r', fontsize=9)
        plt.title('BIPLOT')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
