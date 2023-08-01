import numpy as np
from ..non_linear import barzilai_borwein
from ..linear import BBLS
from scipy.optimize import linprog
import scipy.sparse as sp
from tqdm import tqdm
import pylops

def lagrange_newton_BB(
        f,
        grad_f,
        H_f,
        constraints=[],
        grad_constraints=[],
        H_constraints=[],
        A_eq=None,
        b_eq=None,
        x0=None,
        dim=None,
        dtype=np.float64,
        damp=1,
        inner_iter=20,
        outer_iter=20,
        verbose=False,
        bounds=None,
        callback=None
    ):

    if x0 is None:
        x = np.zeros(dim, dtype=dtype)
    else:
        x = x0

    l = np.zeros(len(constraints))
    if A_eq is not None:
        m = np.zeros(A_eq.shape[0])
    else:
        m = np.array([])

    if verbose:
        loop = tqdm(range(outer_iter))
    else:
        loop = range(outer_iter)

    for i in loop:
        if callback is not None:
            callback(i, x, l, m)

        H = [H_f(x)]
        grad = [grad_f(x)]

        if constraints != []:

            # construct jacobian of the constraints
            J_c = np.vstack([g(x) for g in grad_constraints])

            # expand hessian of the lagrangian
            H[0] += reduce(lambda x,y: x+y,(l[i]*H_constraints[i](x) for i in range(l.size)))
            H.append(J_c)

            # expand gradient of lagrangian
            grad_c = np.array([grad_constraints[i](x) for i in range(l.size)]).T
            grad[0] += grad_c @ l 
            grad.append([c(x) for c in constraints])

        if A_eq is not None:
            # expand hessian of the lagrangian
            H.append(A_eq)

            # expand gradient of lagrangian
            grad[0] += A_eq.T @ m
            grad.append(A_eq @ x - b_eq)

        if len(H) > 1:
            H_c = pylops.VStack(H[1:])
            H = pylops.Block([
                [H[0], H_c.T],
                [ H_c, pylops.Zero(l.size + m.size)]
            ])
        else:
            H = H[0]
        grad = np.concatenate(grad)

        def projector(dxlm):
            dx = dxlm[:len(x)]
            dlm = dxlm[len(x):]

            dx = np.clip(dx, bounds[0]-x, bounds[1]-x)
            return np.concatenate([dx, dlm])

        dx = BBLS(H, -grad, projector=projector, max_iter=inner_iter)

        # update x
        x = x + damp*dx[:len(x)]
        l = l + dx[len(x):len(x)+len(l)]
        m = m + dx[len(x)+len(l):]

    return x


def SLQP_TV_min(
        f,
        grad_f,
        H_f,
        D,
        x0=None,
        dim=None,
        dtype=np.float64,
        damp=1,
        max_iter=20,
        verbose=True,
        callback=None,
    ):
    if x0 is None:
        x = np.zeros(dim + 2*D.shape[0], dtype=dtype)
    else:
        x = np.concatenate([x0, D @ x0])
        dim = len(x0)

    c = np.concatenate([np.zeros(dim), np.ones(D.shape[0]), np.ones(D.shape[0])])
    I = sp.identity(D.shape[0])
    A_eq = sp.bmat([[D, -I, I]])

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        callback(x[:dim], x[dim:dim+D.shape[0]] - x[dim+D.shape[0]:])
        A_ub = np.zeros(len(c))
        A_ub[:dim] = -grad_f(x[:dim])[np.newaxis]
        b_ub = f(x[:dim])
        b_eq = -(A_eq @ x)
        LP_result = linprog(
            c,
            method="highs-ipm",
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(None, None)]*dim + [(-g, None) for g in x[dim:]],
            options={
                "presolve": True,
                "disp": verbose
            }
        )
        #breakpoint()
        d = LP_result.x
        x = x + damp*d/(i+1)
    return x[:dim], x[dim:dim+D.shape[0]] - x[dim+D.shape[0]:]

def subspace_linprog(
        c,
        A,
        b,
        dim=10,
        callback=None,
        max_iter=50,
    ):
    V = golub_kahan(A, b, dim)
    cV = V.T @ c
    AV = np.zeros((A.shape[0], V.shape[1]))
    for i, col in enumerate(V.T):
        AV[:, i] = A @ col
    Q, R = np.linalg.qr(AV)
    result = linprog(
        cV,
        method="revised simplex",
        A_eq=R,
        b_eq=Q.T @ b,
        #bounds=[(None, None)]*len(cV),
        callback=callback,
        options={
            "presolve": False,
            "maxiter": max_iter
        }
    )
    result.x = V @ result.x
    return result


def subspace_newton(
        grad_f,
        H_f,
        x0=None,
        dim=None,
        dtype=np.float64,
        inner_iter=10,
        outer_iter=10,
        verbose=False,
        callback=None
    ):
    x = np.zeros(dim, dtype=dtype)
    res = grad_f(x)
    V = res.reshape(-1, 1)/np.linalg.norm(res)
    y = np.zeros(1, dtype=dtype)

    if verbose:
        loop = tqdm(range(1, outer_iter))
    else:
        loop = range(1, outer_iter)

    for i in loop:
        x = V @ y
        subspace_grad = V.T @ grad_f(x)
        H = H_f(x)
        HV = np.zeros(V.shape)
        for i, col in enumerate(V.T):
            HV[:, i] = H @ col
        subspace_H = V.T @ HV
        y = y - np.linalg.solve(subspace_H, subspace_grad)
        x = V @ y
        res = grad_f(x)
        V = np.hstack([V, res.reshape(-1, 1)/np.linalg.norm(res)])
        y = np.concatenate([y, [0]])
    return x, V


def subspace_barzilai_borwein(
        grad_f,
        x0=None,
        dim=None,
        dtype=np.float64,
        bounds=None,
        projector=None,
        inner_iter=10,
        outer_iter=10,
        verbose=False,
        callback=None
    ):
    x = np.zeros(dim, dtype=dtype)
    res = grad_f(x)
    V = res.reshape(-1, 1)/np.linalg.norm(res)
    y = np.zeros(1, dtype=dtype)
    for i in range(1, outer_iter):
        subspace_grad = lambda y: V.T @ grad_f(V @ y)
        y = barzilai_borwein(subspace_grad, x0=y, max_iter=inner_iter, verbose=verbose)
        x = V @ y
        res = grad_f(x)
        V = np.hstack([V, res.reshape(-1, 1)/np.linalg.norm(res)])
        y = np.concatenate([y, [0]])
    return x, V


def KKTBB(
        f,
        grad_f,
        H_f,
        constraints=[],
        grad_constraints=[],
        H_constraints=[],
        x0=None,
        l0=None,
        m0=None,
        dim=None,
        dtype=np.float64,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    if x0 is None:
        x0 = np.zeros(dim, dtype=dtype)
    dim = len(x0)
    if l0 is None:
        l0 = np.zeros(len(constraints), dtype=dtype)
    if m0 is None:
        m0 = np.zeros(dim, dtype=dtype)

    def grad_L(xlm):
        x = xlm[:dim]
        l = xlm[dim:dim+len(constraints)]
        m = xlm[dim+len(constraints):]

        stat_res = grad_f(x) + sum(l[i] * grad_c(x) for i, grad_c in enumerate(grad_constraints)) - m
        H = H_f(x) + reduce(lambda x,y: x+y, (l[i]*H_constraints[i](x) for i in range(l.size)))
        grad_x = H @ stat_res \
               + sum(grad_c(x) * c(x) for c, grad_c in zip(constraints, grad_constraints)) \
               + m * np.dot(m, x)
        grad_l = np.array([np.dot(grad_c(x), stat_res) for grad_c in grad_constraints])
        grad_m = -stat_res + x * np.dot(m, x)
        return np.concatenate([grad_x, grad_l, grad_m])

    def projector(xlm):
        x = xlm[:dim]
        l = xlm[dim:dim+len(constraints)]
        m = xlm[dim+len(constraints):]

        x = np.clip(x, 0, np.inf)
        m = np.clip(m, 0, np.inf)
        return np.concatenate([x, l, m])

    xlm = barzilai_borwein(
        grad_L,
        x0=np.concatenate([x0, l0, m0]),
        projector=projector,
        max_iter=max_iter,
        verbose=verbose,
        callback=callback
    )
    x = xlm[:dim]
    l = xlm[dim:dim+len(constraints)]
    m = xlm[dim+len(constraints):]
    return x

def LBB(
        f,
        grad_f,
        H_f,
        constraints=[],
        grad_constraints=[],
        H_constraints=[],
        x0=None,
        l0=None,
        dim=None,
        dtype=np.float64,
        max_iter=100,
        verbose=False,
        callback=None
    ):
    if x0 is None:
        x0 = np.zeros(dim, dtype=dtype)
    dim = len(x0)
    if l0 is None:
        l0 = np.zeros(len(constraints), dtype=dtype)

    def grad_L(xl):
        x = xl[:dim]
        l = xl[dim:dim+len(constraints)]

        stat_res = grad_f(x) + sum(l[i] * grad_c(x) for i, grad_c in enumerate(grad_constraints))
        H = H_f(x) + reduce(lambda x,y: x+y, (l[i]*H_constraints[i](x) for i in range(l.size)))
        grad_x = H @ stat_res \
               + sum(grad_c(x) * c(x) for c, grad_c in zip(constraints, grad_constraints))
        grad_l = np.array([np.dot(grad_c(x), stat_res) for grad_c in grad_constraints])
        return np.concatenate([grad_x, grad_l])

    xl = barzilai_borwein(
        grad_L,
        x0=np.concatenate([x0, l0]),
        max_iter=max_iter,
        verbose=verbose,
        callback=callback
    )
    x = xl[:dim]
    l = xl[dim:dim+len(constraints)]
    return x