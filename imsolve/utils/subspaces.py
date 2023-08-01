import numpy as np


def golub_kahan(A, b, dim):
    u = b/np.linalg.norm(b)
    v = np.zeros(A.shape[1])
    norm_w = 0
    V = []
    for _ in range(dim):
        r = A.T @ u - norm_w*v
        norm_r = np.linalg.norm(r)
        v = r/norm_r
        V.append(v)
        w = A @ v - norm_r*u
        norm_w = np.linalg.norm(w)
        u = w/norm_w
    return np.array(V).T