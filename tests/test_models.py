import numpy as np
from scipy.linalg import lstsq
from machine_learning.pca import PCA
from machine_learning.mlr import LinearRegression
from machine_learning.kNN import KNN
from machine_learning.ca import HierarchicalCluster


def test_pca_matches_sklearn_covariance_spectrum():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(40, 5))
    model = PCA(X).fit()
    standardized = (X - X.mean(0)) / X.std(0, ddof=1)
    expected = np.linalg.svd(standardized, compute_uv=False) ** 2 / (len(X)-1)
    assert np.allclose(model.explained_variance_, expected, atol=1e-7)
    assert np.allclose(model.transform().mean(0), 0, atol=1e-8)


def test_linear_regression_coefficients():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 2))
    y = 2.5 + X @ np.array([1.2, -0.7]) + rng.normal(0, .05, 50)
    model = LinearRegression(X, y)
    model.fit()
    expected = lstsq(np.column_stack([np.ones(50), X]), y)[0]
    assert np.allclose(model.beta.ravel(), expected, atol=1e-8)


def test_knn_predictions():
    model = KNN(k=3).fit([[0], [1], [5], [6]], [0, 0, 1, 1])
    assert np.array_equal(model.predict([[.2], [5.8]]), [0, 1])


def test_hierarchical_clustering_merges_all_points():
    model = HierarchicalCluster([np.array([0.]), np.array([1.]), np.array([8.])])
    model.fit()
    assert len(model.history) == 3
    assert len(model.history[-1]) == 1


def test_classical_mds_reproduces_euclidean_distances():
    from machine_learning.mds import MDS
    points = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 2.]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    embedded = MDS(distances).fit(n_components=2)
    reconstructed = np.linalg.norm(
        embedded[:, None, :] - embedded[None, :, :], axis=2
    )
    assert np.allclose(reconstructed, distances, atol=1e-7)
