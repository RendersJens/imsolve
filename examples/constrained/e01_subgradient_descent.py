from autograd import numpy as np
from autograd import grad
from imsolve import subgradient_descent

"""
Test problem:
Minimize x0**2 + x1**2
s.t  x0 * x1 >= 2

solutions: (+-sqrt(2), +-sqrt(2))
"""

def f(x):
    return np.dot(x, x)

def constraint(x):
    return 2 - x[0]*x[1]

def callback(i, x, x_best, flag):
    print(x_best)

x = subgradient_descent(
    f,
    grad(f),
    constraints=[constraint],
    grad_constraints=[grad(constraint)],
    x0=np.ones(2),
    max_iter=100,
    callback=callback
)