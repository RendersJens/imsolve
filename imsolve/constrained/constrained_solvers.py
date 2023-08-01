import numpy as np
import pylops
from tqdm import tqdm
from scipy.sparse.linalg import minres, lsqr
from functools import reduce
from ..non_linear import barzilai_borwein


def PDHG_linprog(
        c, A, b, R, C,
        x0=None,
        l0=None,
        bounds=(0, np.inf),
        max_iter=200,
        verbose=True,
        callback=None
    ):
    # initialize unknowns
    if x0 is None:
        x = np.zeros(A.shape[1])
    else:
        x = x0

    # initialize lagrange multipliers
    if l0 is None:
        l = np.zeros(A.shape[0])
    else:
        l = l0

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(i, x, l)

        xp = x
        x = x - C @ (A.T @ l + c)
        x = np.clip(x, *bounds)
        l = l + R @ (A @ (2*x - xp) - b)

    return x


def augmented_lagrangian(
        grad_f,
        eq=None,
        ineq=None,
        vjp_eq=None,
        vjp_ineq=None,
        bounds=None,
        x0=None,
        l0=None,
        m=0.01,
        dim=None,
        inner_iter=30,
        inner_options={},
        max_iter=10,
        m_mult=2
    ):

    # initialize unknowns
    if x0 is None:
        x = np.zeros(dim)
    else:
        x = x0

    # initialize lagrange multipliers
    if l0 is None:
        l = np.ones(len(eq(x)))*0.001
    else:
        l = l0

    # gradient of augmented lagrangian with respect to x
    def grad_L(x):
        grad = grad_f(x)
        if eq is not None:
            grad = grad + vjp_eq(x, l + m*eq(x))
        # print(np.linalg.norm(grad))
        return grad

    for i in tqdm(range(max_iter)):
        print(eq(x))
        x = barzilai_borwein(grad_L, x0=x, bounds=bounds, max_iter=inner_iter, **inner_options)
        l += m*eq(x)
        m *= m_mult

    return x


def subgradient_descent(
        f,
        grad_f,
        constraints=None,
        grad_constraints=None,
        x0=None,
        dim=None,
        obj_scale=1,
        dtype=np.float64,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    """ subgradient descent with Polyak step size and inequality constraints

        For to enforce the constraint g(x) <= c you need to add
        g(x) - c to the constraints list, and the subgradient of
        g(x) - c to the grad_constraints list.

        x is the new solution
        xp is the previous solution
    """

    if x0 is None:
        x = np.zeros(dim, dtype = dtype)
    else:
        x = x0

    x_best = None
    f_best = None
    flag = -1

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        if callback is not None:
            callback(i, x, x_best, flag)

        # this flag tells us if we need to work on a constraint
        # or on the main objective. If -1, all constraints are met.
        # otherwise it is the index of an unmet constraint.
        flag = -1
        if constraints is not None:
            for j, constraint in enumerate(constraints):
                if constraint(x) > 0:
                    flag = j
                    break
        if flag == -1:
            if x_best is None or f(x) < f_best:
                x_best = x
                f_best = f(x)

        # calculate gradient
        if flag == -1:
            grad = grad_f(x)
        else:
            grad = grad_constraints[flag](x)

        # Polyak step size
        if flag == -1:
            if f_best is not None:
                a = (f(x) - f_best + obj_scale/(i+1))/np.linalg.norm(grad)**2
            else:
                a = (obj_scale/(i+1))/np.linalg.norm(grad)**2
        else:
            a = (max(constraints[flag](x), 0) + obj_scale/(i+1))/np.linalg.norm(grad)**2

        # new solution: gradient descent step
        xp = x
        x = x - a*grad

    return x_best


def lagrange_newton_krylov(
        f,
        grad_f,
        H_f,
        constraints=[],
        grad_constraints=[],
        H_constraints=[],
        x0=None,
        dim=None,
        dtype=np.float64,
        damp = 1,
        lin_solver = "MINRES",
        max_iter=100,
        inner_iter=20,
        verbose=False,
        bounds=None,
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

        if constraints != []:

            # construct jacobian of the constraints
            J_c = np.vstack([g(x) for g in grad_constraints])

            # construct full hessian of the lagrangian
            H = H_f(x) + reduce(lambda x,y: x+y,(l[i]*H_constraints[i](x) for i in range(l.size)))
            H = pylops.Block([[H  , J_c.T              ],
                              [J_c, pylops.Zero(l.size)]])

            # construct gradient of lagrangian
            grad_c = np.array([grad_constraints[i](x) for i in range(l.size)]).T
            grad = np.concatenate([grad_f(x) + grad_c @ l] + [[c(x)] for c in constraints])

        else:
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

        # update x
        x = x + damp*dx[:len(x)]
        l = l + damp*dx[len(x):]

    return x
