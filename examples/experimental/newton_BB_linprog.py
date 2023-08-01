import sys
sys.path.insert(1, '..')

import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from scipy.optimize import linprog
import scipy.sparse as sp
from imsolve.experimental import lagrange_newton_BB

# create phantom
n_pix = 128
phantom = shepp2d(n_pix)[0]/255

# create projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
angles = np.linspace(0,np.pi,30)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, angles)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
A = astra.OpTomo(proj_id)
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
N1 = pylops.Zero(A.shape[0], I_grad.shape[1])
N2 = pylops.Zero(grad.shape[0], I_proj.shape[1])
N3 = pylops.Zero(1, grad.shape[1])
N4 = pylops.Zero(1, I_grad.shape[1])
A_eq = pylops.Block([
        [   A,      N1,     N1, -I_proj,  I_proj],
        [grad, -I_grad, I_grad,      N2,      N2],
        [  N3,      N4,     N4, one_res/len(angles), one_res/len(angles)]
])

b_eq = np.concatenate([b, np.zeros(2*im_size), [tol/len(angles)]])

f = lambda x: 1/2*np.dot(c, x)**2/1000000
grad_f = lambda x: c*np.dot(c, x)/1000000
H_f = lambda x: (1/1000000)*pylops.MatrixMult(c[np.newaxis]).T @ pylops.MatrixMult(c[np.newaxis])

# f = lambda x: np.dot(c, x)
# grad_f = lambda x: c
# H_f = lambda x: pylops.Zero(len(c))

lb = np.zeros(len(c))
lb[:n_pix**2] = -np.inf

x = lagrange_newton_BB(
    f,
    grad_f,
    H_f,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=(lb, np.inf),
    #x0=np.load("x.npy"),
    dim=A_eq.shape[1],
    damp=0.5,
    outer_iter=2000,
    inner_iter=10,
    verbose=True,
    callback=lambda i, x, l, m: print(f(x), np.linalg.norm(A_eq @ x - b_eq))
)
rec = x[:im_size]
rec_grad_p = x[im_size:3*im_size]
rec_grad_m = x[3*im_size:5*im_size]
rec_grad = rec_grad_p - rec_grad_m
rec_res_p = x[5*im_size:5*im_size+proj_size]
rec_res_m = x[5*im_size+proj_size:]
rec_res = rec_res_p - rec_res_m

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