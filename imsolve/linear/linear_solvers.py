import numpy as np
import pylops
from tqdm import tqdm


def BBLS(
        A,
        b,
        x0=None,
        norm=None,
        bounds=None,
        projector=None,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    """ Projected gradient descent with BB step size and bounds
        This functional is minimized
        f(x) = 1/2 ||Ax-b||^2
        the norm is the matrix norm of the passed matrix.
        None means euclidean norm

        x is the new solution
        xp is the previous solution
    """

    if norm is None:
        N = pylops.Identity(A.shape[0])
    else:
        N = norm

    if x0 is None:
        x = np.zeros(A.shape[1], dtype = A.dtype)
    else:
        x = x0

    # setup for the first iteration:

    # gradient at begin position
    grad = A.T @ (N @ (A @ x - b))

    # we start with a step of unit length
    a = a2 = 1/np.linalg.norm(grad)

    # gradient descent step
    xp = x
    x = x - a*grad

    # apply the bounds
    if not bounds is None:
        x = np.clip(x, *bounds)

    # apply the projector
    if not projector is None:
        x = projector(x)

    # now that we have our previous values xp and xpp
    # we can start iterating
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x, stepsize=a, stepsize2=a2)

        res = A @ x - b

        # gradient at previous solution
        gradp = grad
        grad = A.T @ N @ res

        # BB step size
        a = np.dot(x-xp, grad - gradp)/np.dot(grad-gradp, grad - gradp)
        a2 = np.dot(x-xp, x - xp)/np.dot(x-xp, grad - gradp)

        # new solution: gradient descent step
        xp = x
        x = x - a*grad

        # apply the bounds
        if not bounds is None:
            x = np.clip(x, *bounds)

        # apply the projector
        if not projector is None:
            x = projector(x)

    return x


def CGLS(
        A,
        b, 
        recursive=True,
        dtype=np.float64,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    x = np.zeros(A.shape[1], dtype=dtype)

    # residue at begin position
    res = b - A @ x

    # gradient at begin position
    grad = A.T @ res

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x)

        # update
        grad2 = np.dot(grad,grad)
        if i == 0:
            p = grad
        else:
            p = grad + grad2/grad2p * p
        Ap = A @ p
        a = grad2/np.dot(Ap, Ap)
        x = x + a*p
            
        if recursive:
            res -= a*Ap
        else:
            res = b - A @ x # slower but better numerical stability
        grad2p = grad2
        grad = A.T @ res

    return x


def CG(
        A,
        b,
        dtype=np.float64,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    x = np.zeros(A.shape[1], dtype=dtype)

    # residue at begin position
    res = b - A @ x

    # conjugate gradient at begin position
    p = res

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x)

        # update
        res2 = np.dot(res, res)
        Ap = A @ p
        a = res2/np.dot(p, Ap)
        x = x + a*p
        res = res - a*Ap
        p = res + np.dot(res, res)/res2 * p

    return x


def SIRT(
        A,
        b,
        x0=None,
        C=None,
        R=None,
        dtype=np.float64,
        bounds=None,
        damp=1,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    if x0 is None:
        x = np.zeros(A.shape[1], dtype=dtype)
    else:
        x = x0

    if R is None:
        rowsums = A @ np.ones(A.shape[1], dtype=dtype)
        rowsums[rowsums<1e-10] = np.inf
        irowsums = 1/rowsums
        R = pylops.Diagonal(irowsums, dtype=dtype)
    if C is None:
        colsums = A.T @ np.ones(A.shape[0], dtype=dtype)
        C = pylops.Diagonal(1/colsums, dtype=dtype)

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x)

        # update
        res = A @ x - b
        x = x - damp * (C @ (A.T @ (R @ res)))
        if bounds is not None:
            x = np.clip(x, *bounds)

    return x