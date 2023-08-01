import sys
sys.path.insert(1, '..')

import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from scipy.optimize import linprog
from imsolve.experimental import subspace_linprog
from imsolve.utils import golub_kahan
import scipy.sparse as sp

# create phantom
n_pix = 32
dim = 500
phantom = shepp2d(n_pix)[0]/255

# create projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, np.linspace(0,np.pi,100))
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
A = astra.OpTomo(proj_id)
A = 1/n_pix * A
im_size = A.shape[1]
proj_size = A.shape[0]

# create sino
I_0 = 1e4
p = I_0 * np.exp(-A @ phantom.ravel())
p = np.random.poisson(p)/I_0

# log normalize
b = -np.log(p)

V = golub_kahan(A, b, dim)
AV = np.zeros((A.shape[0], V.shape[1]))
for i, col in enumerate(V.T):
    AV[:, i] = A @ col

# create identity operators
I_im_1d = sp.identity(n_pix)
I_im = sp.identity(im_size)
I_dim = sp.identity(dim)
I_grad = sp.identity(2*im_size)
I_proj = sp.identity(proj_size)

# create image gradient operator
grad_1d = np.ones((2, im_size))
grad_1d[0] *= -1
grad_1d = sp.spdiags(grad_1d, (0,1), n_pix, n_pix)
D = sp.vstack([
    sp.kron(grad_1d, I_im_1d),
    sp.kron(I_im_1d, grad_1d)
])
DV = np.zeros((D.shape[0], V.shape[1]))
for i, col in enumerate(V.T):
    DV[:, i] = D @ col

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

A_eq = sp.bmat([
        [AV,    None, None, -I_proj, I_proj],
        [DV, -I_grad, I_grad,  None,   None]
])

b_eq = np.concatenate([b, np.zeros(2*im_size)])

c = np.concatenate([
        np.zeros(dim),
        np.ones(2*im_size),
        np.ones(2*im_size),
        np.zeros(proj_size),
        np.zeros(proj_size)
    ])

A_ub = np.concatenate([
        np.zeros(dim),
        np.zeros(2*im_size),
        np.zeros(2*im_size),
        np.ones(proj_size),
        np.ones(proj_size)
    ])[np.newaxis]

b_ub = np.array([tol])

# A_ub = sp.lil_matrix((V.shape[0]+1, len(c)))
# A_ub[:-1, :dim] = -V
# A_ub[-1] = np.concatenate([
#         np.zeros(dim),
#         np.zeros(2*im_size),
#         np.zeros(2*im_size),
#         np.ones(proj_size),
#         np.ones(proj_size)
#     ])[np.newaxis]

# b_ub = np.zeros(V.shape[0]+1)
# b_ub[-1] = tol

# one_res = sp.coo_matrix(np.ones((1, proj_size)))
# A_eq = sp.bmat([
#         [AV,    None,   None, -I_proj,  I_proj],
#         [DV, -I_grad, I_grad,    None,    None],
#         [None,    None,   None, one_res, one_res]
# ])



#A_eq = pylops.VStack([A_eq])

# b_eq = np.concatenate([b, np.zeros(2*im_size), [tol]])

def callback(result):
        print(result)


result = linprog(c, A_eq=A_eq, b_eq=b_eq,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(None, None)]*dim + [(0, None)]*(len(c)-dim),
    method="highs-ipm",
    options={
        "presolve": False,
        "disp": True
    }
)

print(result)
rec = V @ result.x[:dim]
rec_grad_p = result.x[dim:dim+2*im_size]
rec_grad_m = result.x[dim+2*im_size:dim+4*im_size]
rec_grad = rec_grad_p - rec_grad_m

plt.figure()
plt.imshow(rec.reshape(n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.figure()
plt.imshow((D @ rec).reshape(2*n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.figure()
plt.imshow(rec_grad.reshape(2*n_pix, n_pix), cmap="gray")
plt.colorbar()

plt.show()