import numpy as np

from machine_learning.linalg import SVD

A = np.array([[2, 1, 0, 4], [1, 1, 3, 7],
                [0, 3, 7, 2], [4, 7, 2, 5]], dtype=float)
U, S, V = SVD(A).decompose()
print("Reconstruction error:", np.linalg.norm(A - U @ np.diag(S) @ V.T))