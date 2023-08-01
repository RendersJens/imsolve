import numpy as np


def operator_inf_norm(A, dtype=np.float64):
    """
    Calculate the inf-norm of a linear operator
    that only has positive coefficients
    """
    x = np.ones(A.shape[1], dtype=dtype)
    return (A @ x).max()


def operator_1norm(A, dtype=np.float64):
    """
    Calculate the 1-norm of a linear operator
    that only has positive coefficients
    """
    x = np.ones(A.T.shape[1], dtype=dtype)
    return (A.T @ x).max()


def operator_2norm(A, max_iter, dtype=np.float64):
    """
    Calculate the 2-norm of a linear operator
    """
    B = A.T @ A
    b = np.random.normal(0,1,B.shape[1]).astype(dtype)
    b = b/np.linalg.norm(b)
    for i in range(max_iter):
        b = B @ b
        b = b/np.linalg.norm(b)
    return np.sqrt(np.linalg.norm(B@b)/np.linalg.norm(b))