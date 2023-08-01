import numpy as np
from scipy.sparse.linalg import lsqr


def dykstra(x, proj1, proj2, max_iter):
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    for i in range(max_iter):
        y = proj1(x + p)
        p = x + p - y
        x = proj2(y + q)
        q = y + q - x
    return x

def project_linear(x, A, b):
    return x - A.T @ np.linalg.solve(A @ A.T, A @ x - b)

def project_linear_iterative(x, A, b, max_iter=20):
    return x - A.T @ lsqr(A @ A.T, A @ x - b, iter_lim=max_iter)[0]