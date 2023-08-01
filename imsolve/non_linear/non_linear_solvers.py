import numpy as np
import pylops
from tqdm import tqdm
from scipy.sparse.linalg import minres, lsqr
from collections import deque
from ..utils.line_search import armijo_line_search


def barzilai_borwein(
        grad_f,
        x0=None,
        dim=None,
        dtype=np.float64,
        bounds=None,
        projector=None,
        prox=None,
        a_init=1,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    """ Projected gradient descent with BB step size and bounds

        x is the new solution
        xp is the previous solution
    """

    if x0 is None:
        x = np.zeros(dim, dtype = dtype)
    else:
        x = x0

    # setup for the first iteration:

    # gradient at begin position
    grad = grad_f(x)

    # we start with a step of unit length
    a = a_init/np.linalg.norm(grad)

    # gradient descent step
    xp = x
    x = x - a*grad

    # apply the bounds
    if bounds is not None:
        x = np.clip(x, *bounds)

    # apply the projector
    if projector is not None:
        x = projector(x)

    # apply the proxmal operator
    if prox is not None:
        x = prox(x, a)

    # now that we have our previous values xp and xpp
    # we can start iterating
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x)

        # gradient at previous solution
        gradp = grad
        grad = grad_f(x)

        # BB step size
        y = grad - gradp
        norm_y_squared = np.dot(y, np.conj(y))
        if norm_y_squared > 0:
            s = x-xp
            a = abs(np.dot(s, np.conj(y)))/norm_y_squared
        else:
            print("exited BB early")
            break

        # new solution: gradient descent step
        xp = x
        x = x - a*grad

        # apply the bounds
        if bounds is not None:
            x = np.clip(x, *bounds)

        # apply the projector
        if projector is not None:
            x = projector(x)

        # apply the proxmal operator
        if prox is not None:
            x = prox(x, a)

    return x


def gradient_descent(
        grad_f,
        x0=None,
        dim=None,
        dtype=np.float64,
        bounds=None,
        stepsize=1,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    """ Projected gradient descent
    """

    if x0 is None:
        x = np.zeros(dim, dtype = dtype)
    else:
        x = x0

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(iteration=i, current_solution=x)

        # gradient
        grad = grad_f(x)

        # new solution: gradient descent step
        x = x - stepsize*grad

        # apply the bounds
        if not bounds is None:
            x = np.clip(x, *bounds)

    return x


def nl_conjugate_gradient(
        f,
        grad_f,
        x0 = None,
        dim = None,
        dtype=np.float64,
        damp=1,
        max_iter=100,
        verbose=False,
        callback=None
    ):

    if x0 is None:
        x = np.zeros(dim, dtype=dtype)
    else:
        x = x0

    # setup for the first iteration:

    # gradient at begin position
    grad = grad_f(x)

    # we start with a step along the gradient direction
    d = -grad
    a = armijo_line_search(f, f(x), grad, x, d, verbose=True)

    # conjugate gradient step
    xp = x
    x = x + a*d

    # now that we have our previous value xp
    # we can start iterating
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(x)

        # gradient at previous solution
        gradp = grad
        grad = grad_f(x)

        # Polak-Ribiere formula with automatic reset
        beta = np.dot(grad, grad - gradp)/np.dot(gradp, gradp)
        beta = max(0, beta)

        # conjugate direction update
        dp = d
        d = -grad + beta * dp
        a = armijo_line_search(f, f(x), grad, x, d, verbose=False)

        # new solution: conjugate gradient step
        xp = x
        x = x + a*d

    return x


def newton_krylov(
        f,
        grad_f,
        H_f,
        x0=None,
        dim=None,
        dtype=np.float64,
        damp=1,
        line_search=True,
        lin_solver="MINRES",
        max_iter=100,
        inner_iter=20,
        verbose=False,
        callback=None
    ):

    if x0 is None:
        x = np.zeros(dim, dtype=dtype)
    else:
        x = x0

    l = np.zeros(len(constraints))

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(i, x, l)

        H = H_f(x)
        grad = grad_f(x)

        # calculate newton step
        if lin_solver == "MINRES":
            dx = minres(H, -grad, maxiter=inner_iter)[0]
        elif lin_solver == "CG":
            dx = CG(H, -grad, max_iter=inner_iter)[0]
        elif lin_solver == "CGLS":
            dx = CGLS(H, -grad, max_iter=inner_iter)[0]
        elif lin_solver == "LSQR":
            dx = lsqr(H, -grad, iter_lim=inner_iter)[0]
        else:
            raise NotImplementedError("Linear solver not recognised")

        if line_search:
            damp = armijo_line_search(f, f(x), grad, x, dx, verbose=False)

        # update x
        x = x + damp*dx

    return x


def quasi_newton(
        f,
        grad_f,
        x0=None,
        dim=None,
        Hi0=None,
        memory=None,
        H_strat="SR1",
        dtype=np.float64,
        damp=1,
        line_search=True,
        max_iter=100,
        verbose=False,
        callback=None
    ):

    if x0 is None:
        x = np.zeros(dim, dtype=dtype)
    else:
        x = x0

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    grad = grad_f(x)

    # starting estimated inverse hessian
    if Hi0 is None:
        Hi0 = pylops.Identity(len(x))*(1/np.linalg.norm(grad))

    Hi_list = deque(maxlen=memory)
    for i in loop:
        if callback is not None:
            callback(i, x)

        # Calculate inverse hessian based on remembered information
        Hi = Hi0
        for term in Hi_list:
            Hi = Hi + term

        dx = damp * (Hi @ -grad)

        if line_search:
            damp = armijo_line_search(f, f(x), grad, x, dx, verbose=False)

        # update x
        x = x + damp*dx

        # construct gradient of lagrangian
        gradp = grad
        grad = grad_f(x)

        y = grad - gradp
        if H_strat == "BFGS":
            Hi_list.append(BFGS_Hi_update(Hi, dx, y))
        elif H_strat == "SR1":
            Hi_list.append(SR1_Hi_update(Hi, dx, y))
        elif H_strat == "GoodBroyden":
            Hi_list.append(GoodBroyden_Hi_update(Hi, dx, y))
        elif H_strat == "BadBroyden":
            Hi_list.append(BadBroyden_Hi_update(Hi, dx, y))
        else:
            raise NotImplementedError("Unknown Hessian update strategy")

    return x



#------------------------------------------------------------------------------------------------------
# Quasi-Newton Hessian updaters
#------------------------------------------------------------------------------------------------------
def BFGS_Hi_update(Hi, s, y):
    sy = np.dot(s,y)
    s_op = pylops.MatrixMult(s[np.newaxis])
    Hiy = Hi @ y
    Hiy_op = pylops.MatrixMult(Hiy[np.newaxis])

    # BFGS formula
    return (s_op.T @ s_op) * ((sy + np.dot(y, Hiy))/sy**2) - (Hiy_op.T @ s_op + s_op.T @ Hiy_op)*(1/sy)

def SR1_Hi_update(Hi, s, y):
    z = s - Hi@y
    z_op = pylops.MatrixMult(z[np.newaxis])

    # SR1 formula
    return (z_op.T @ z_op)*(1/np.dot(z, y))

def GoodBroyden_Hi_update(Hi, s, y):
    z = s - Hi @ y
    z_op = pylops.MatrixMult(z[np.newaxis])
    His = Hi.T @ s
    His_op = pylops.MatrixMult(His[np.newaxis])

    # Good Broyden formula
    return (z_op.T @ His_op)*(1/np.dot(His,y))

def BadBroyden_Hi_update(Hi, s, y):
    z = s - Hi @ y
    z_op = pylops.MatrixMult(z[np.newaxis])
    y_op = pylops.MatrixMult(y[np.newaxis])

    # Bad Broyden formula
    return (z_op.T @ y_op)*(1/np.dot(y,y))

def BFGS_H_update(H, s, y):
    y_op = pylops.MatrixMult(y[np.newaxis])
    Hs = H @ s
    Hs_op = pylops.MatrixMult(Hs[np.newaxis])

    return (y_op.T @ y_op)*(1/np.dot(y,s)) - (Hs_op.T @ Hs_op)*(1/np.dot(s, Hs))

def SR1_H_update(H, s, y):
    z = y - H@s
    z_op = pylops.MatrixMult(z[np.newaxis])

    # SR1 formula
    return (z_op.T @ z_op)*(1/np.dot(z, s))
