from sklearn.datasets import fetch_california_housing
from mlr import LinearRegression

data = fetch_california_housing()
X, y = data.data, data.target

regression = LinearRegression(X, y)

regression.fit()
regression.summary()