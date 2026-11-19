from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

from machine_learning.mlr import LinearRegression

data = fetch_california_housing()
X, y = data.data, data.target

idx = data.feature_names.index("AveBedrms")
X = np.delete(X, idx, axis=1)

X = StandardScaler().fit_transform(X)

regression = LinearRegression(X, y)
regression.fit()
regression.summary()
regression.plot_diagnostics()