import numpy as np
from scipy.linalg import qr, svd
from machine_learning.linalg import QRDecomposition, SVD


def test_householder_qr_matches_numpy():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 4))
    decomposition = QRDecomposition(A)
    Q, R = decomposition.get_Q(), decomposition.get_R()
    assert np.allclose(Q @ R, A, atol=1e-9)
    assert np.allclose(Q.T @ Q, np.eye(8), atol=1e-9)
    assert np.allclose(R, np.triu(R))


def test_qr_solve_matches_numpy():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(10, 4))
    b = rng.normal(size=(10, 2))
    actual = QRDecomposition(A).solve(A.T @ b)
    expected = np.linalg.solve(A.T @ A, A.T @ b)
    assert np.allclose(actual, expected, atol=1e-8)


def test_jacobi_svd_matches_scipy_reconstruction():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(7, 4))
    U, s, V = SVD(A).decompose()
    assert np.allclose(U @ np.diag(s) @ V.T, A, atol=1e-8)
    assert np.allclose(s, svd(A, compute_uv=False), atol=1e-8)
