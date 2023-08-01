import numpy as np
import pylops


def eps_norm(x, eps):
    return np.sum(np.sqrt(x**2 + eps))

def d_eps_norm(x, eps):
    return x/np.sqrt(x**2 + eps)

def H_eps_norm(x, eps):
    return pylops.Diagonal(eps/np.sqrt(x**2+eps)**3)


def eps_norm_2(x, eps):
    x1, x2 = np.split(x, 2)
    return np.sum(np.sqrt(x1**2 + x2**2 + eps))

def d_eps_norm_2(x, eps):
    x1, x2 = np.split(x, 2)
    return np.concatenate([x1/np.sqrt(x1**2 + x2**2 + eps),
                           x2/np.sqrt(x1**2 + x2**2 + eps)])


def complex_eps_norm(x, eps):
    x1 = np.real(x)
    x2 = np.imag(x)
    return np.sum(np.sqrt(x1**2 + x2**2 + eps))

def d_complex_eps_norm(x, eps):
    x1 = np.real(x)
    x2 = np.imag(x)
    return x1/np.sqrt(x1**2 + x2**2 + eps) + 1j * x2/np.sqrt(x1**2 + x2**2 + eps)