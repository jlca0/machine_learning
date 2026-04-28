import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from linalg.linalg import QRDecomposition

class LinearRegression:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        self.n, self.k = X.shape
        X_design = np.column_stack([np.ones(self.n), X])
        self.X = QRDecomposition(X_design)
        self.y = y.reshape(-1, 1) if y.ndim == 1 else y
    
    def fit(self) -> None:
        """
        Fits a multiple linear model for the data, ie, computes the regression coefficients, their
        variance matrix and the corresponding residuals.
        """
        cond = self.X.get_condition_number()
        if cond > 40:
            print(f"Advertencia: número de condición alto ({cond:.2e}), posible colinealidad.")
        Qty = self.X.apply_Qt(self.y)
        self.beta = self.X.solve(Qty[:self.k+1, :])
        self.variance = self.X.solve(np.eye(self.k+1))
        self.residuals = self.y - self.X.get_Q()[:, :self.k+1] @ self.X.apply_Qt(self.y)[:self.k+1, :]
    
    def summary(self) -> None:
        cond = self.X.get_condition_number()
        RSS = float(self.residuals.flatten() @ self.residuals.flatten())
        TSS = float(sum((self.y - self.y.mean()) ** 2))
        sigma2 = RSS / (self.n - self.k - 1)

        R2 = 1 - RSS / TSS
        R2_adj = 1 - (1 - R2) * (self.n - 1) / (self.n - self.k - 1)

        se = np.sqrt(np.diag(sigma2 * self.variance))
        t_stats = self.beta.flatten() / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=self.n - self.k - 1))

        F_stat = ((TSS - RSS) / self.k) / (sigma2)
        p_F = 1 - stats.f.cdf(F_stat, self.k, self.n - self.k - 1)

        print(f"R²: {R2:.4f}  |  R² adj: {R2_adj:.4f}  |  cond: {cond:.2e}")
        print(f"σ²: {sigma2:.4f}  |  F: {F_stat:.4f}  (p={p_F:.4f})")
        print(f"\n{'':>12} {'coef':>10} {'se':>10} {'t':>10} {'p':>10}")
        names = ["intercept"] + [f"x{i}" for i in range(1, self.k + 1)]
        for name, b, s, t, p in zip(names, self.beta.flatten(), se, t_stats, p_values):
            print(f"{name:>12} {b:>10.4f} {s:>10.4f} {t:>10.4f} {p:>10.4f}")

    def plot_diagnostics(self) -> None:
        y_hat = (self.y - self.residuals).flatten()
        res = self.residuals.flatten()
        sigma2 = float(res @ res) / (self.n - self.k - 1)
        Q = self.X.get_Q()[:, :self.k+1]
        leverage = np.sum(Q**2, axis=1)
        cooks = (res**2 / ((self.k + 1) * sigma2)) * (leverage / (1 - leverage)**2)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].scatter(y_hat, res, alpha=0.6)
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_xlabel("Valores ajustados")
        axes[0, 0].set_ylabel("Residuos")
        axes[0, 0].set_title("Residuos vs Valores ajustados")

        stats.probplot(res, plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot")

        axes[1, 0].scatter(range(self.n), res, alpha=0.6)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel("Índice")
        axes[1, 0].set_ylabel("Residuos")
        axes[1, 0].set_title("Residuos vs Índice")

        axes[1, 1].stem(range(self.n), cooks)
        threshold = stats.chi2.ppf(0.5, df=self.k+1)
        axes[1, 1].axhline(threshold, color='red', linestyle='--', label=f"Chi²(k+1) mediana = {threshold:.3f}")
        axes[1, 1].set_xlabel("Índice")
        axes[1, 1].set_ylabel("Distancia de Cook")
        axes[1, 1].set_title("Distancia de Cook")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()