import sys
sys.path.insert(1, '..')

import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from scipy.optimize import linprog
import scipy.sparse as sp
import cplex

# create phantom
n_pix = 128
phantom = shepp2d(n_pix)[0]/255

# create projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
angles = np.linspace(0,np.pi,30)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, angles)
proj_id = astra.create_projector('linear', proj_geom, vol_geom)
A = astra.matrix.get(astra.projector.matrix(proj_id))
A = 1/n_pix * A
im_size = A.shape[1]
proj_size = A.shape[0]

# create identity operators
I_im_1d = sp.identity(n_pix)
I_im = sp.identity(im_size)
I_grad = sp.identity(2*im_size)
I_proj = sp.identity(proj_size)

# create image gradient operator
grad_1d = np.ones((2, im_size))
grad_1d[0] *= -1
grad_1d = sp.spdiags(grad_1d, (0,1), n_pix, n_pix)
grad = sp.vstack([
    sp.kron(grad_1d, I_im_1d),
    sp.kron(I_im_1d, grad_1d)
])

# create sino
I_0 = 1e5
p = I_0 * np.exp(-A @ phantom.ravel())
p = np.random.poisson(p)/I_0

# log normalize
b = -np.log(p)

# check tolerance
tol = np.linalg.norm(A @ phantom.ravel() - b, 1)

# define linear program

# c = np.concatenate([
#         np.zeros(im_size),
#         np.ones(2*im_size),
#         np.ones(2*im_size),
#         np.zeros(proj_size),
#         np.zeros(proj_size)
#     ])

# A_ub = np.concatenate([
#         np.zeros(im_size),
#         np.zeros(2*im_size),
#         np.zeros(2*im_size),
#         np.ones(proj_size),
#         np.ones(proj_size)
#     ])[np.newaxis]

# b_ub = np.array([tol])

# A_eq = sp.bmat([
#         [   A,  None, None, -I_proj, I_proj],
#         [grad, -I_grad, I_grad,    None,   None]
# ])

# b_eq = np.concatenate([b, np.zeros(2*im_size)])

c = np.concatenate([
        np.zeros(im_size),
        np.ones(2*im_size),
        np.ones(2*im_size),
        np.zeros(proj_size),
        np.zeros(proj_size)
    ])

one_res = sp.coo_matrix(np.ones((1, proj_size)))
A_eq = sp.bmat([
        [   A,    None,   None, -I_proj,  I_proj],
        [grad, -I_grad, I_grad,    None,    None],
        [None,    None,   None, one_res, one_res]
])

b_eq = np.concatenate([b, np.zeros(2*im_size), [tol]])

# lp = cplex.Cplex()
# lp.objective.set_sense(lp.objective.sense.minimize)
# lp.linear_constraints.add(
#     rhs=np.concatenate([b_eq, b_ub]),
#     senses="E"*len(b_eq) + "L"*len(b_ub),
# )
# lp.variables.add(
#     obj=c,
#     lb=[-cplex.infinity]*n_pix**2 + [0]*(len(c)-n_pix**2),
#     ub=[cplex.infinity]*len(c)
# )
# coeffs = sp.vstack([A_eq, A_ub]).tocoo()

lp = cplex.Cplex()
lp.objective.set_sense(lp.objective.sense.minimize)
lp.linear_constraints.add(
    rhs=b_eq,
    senses="E"*len(b_eq),
)
lp.variables.add(
    obj=c,
    lb=[-cplex.infinity]*n_pix**2 + [0]*(len(c)-n_pix**2),
    ub=[cplex.infinity]*len(c)
)
coeffs = A_eq.tocoo()

rows = [int(i) for i in coeffs.row]
cols = [int(i) for i in coeffs.col]
vals = coeffs.data

lp.linear_constraints.set_coefficients(zip(rows, cols, vals))

alg = lp.parameters.lpmethod.values
lp.parameters.lpmethod.set(alg.barrier)
lp.solve()

print(lp.solution.get_status())
x = np.array(lp.solution.get_values())
np.save("x", x)
print(np.dot(c, x))
rec = x[:im_size]
rec_grad_p = x[im_size:3*im_size]
rec_grad_m = x[3*im_size:5*im_size]
rec_grad = rec_grad_p - rec_grad_m

plt.figure()
plt.imshow(rec.reshape(n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.figure()
plt.imshow((grad @ rec).reshape(2*n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.figure()
plt.imshow(rec_grad.reshape(2*n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.show()