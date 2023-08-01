import numpy as np


def armijo_line_search(
        f,
        fx,
        grad,
        x,
        d,
        t=0.5,
        c=1e-4,
        a=1.0,
        max_iter=20,
        verbose=False
    ):
    m = np.dot(grad, d/np.linalg.norm(d))
    iterations = 0
    new_f = f(x + a*d)
    while iterations < max_iter and new_f > fx + a*c*m:
        if verbose:
            print(iterations, new_f, fx + a*c*m)
        a *= t
        new_f = f(x + a*d)
        iterations += 1
    return a