import sys
sys.path.insert(1, '..')

import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from scipy.optimize import linprog
from imsolve.constrained import PDHG_linprog
import scipy.sparse as sp

# create phantom
n_pix = 128
phantom = shepp2d(n_pix)[0]/255

# create projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, np.linspace(0,np.pi,30))
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

R = abs(A_eq) @ np.ones(A_eq.shape[1])
R = pylops.Diagonal(1/R)
C = abs(A_eq.T) @ np.ones(A_eq.shape[0])
C = pylops.Diagonal(1/C)

# result = linprog(c, A_eq=A_eq, b_eq=b_eq,
#     A_ub=A_ub,
#     b_ub=b_ub,
#     method="highs-ipm",
#     bounds=[(None, None)]*n_pix**2 + [(0, None)]*(len(c)-n_pix**2),
#     #callback=callback,
#     options={
#         "presolve": True,
#         "disp": True
#     }
# )
# print(result)
# rec = result.x[:im_size]
# rec_grad_p = result.x[im_size:3*im_size]
# rec_grad_m = result.x[3*im_size:5*im_size]
# rec_grad = rec_grad_p - rec_grad_m
def callback(i, x, l):
    print(np.linalg.norm(A @ x[:im_size] - b, 1))

lb = np.zeros(len(c))
lb[:n_pix**2] = -np.inf
x = PDHG_linprog(
    c, A_eq, b_eq, R, C,
    bounds=(lb, np.inf),
    max_iter=100_000,
    verbose=False,
    callback=callback
)
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