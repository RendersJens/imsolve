import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
import scipy.sparse as sp
from imsolve import barzilai_borwein
from imsolve.experimental import SLQP_TV_min

# create phantom
n_pix = 256
phantom = shepp2d(n_pix)[0]/255

# creat CT projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
angles = np.linspace(0,np.pi,300)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, angles)
proj_id = astra.create_projector('linear', proj_geom, vol_geom)
A = astra.optomo.OpTomo(proj_id)
A = 1/n_pix * pylops.VStack([A])

# simulate CT scan
I_0 = 1e5
p = I_0 * np.exp(-A @ phantom.ravel())
p = np.random.poisson(p)/I_0

# log normalize
b = -np.log(p)

tol = 1/2*np.linalg.norm(A @ phantom.ravel() - b)**2

# solve
f = lambda x: tol - 1/2*np.linalg.norm(A @ x - b)**2
grad_f = lambda x: -A.T @ (A @ x - b)
H_f = lambda x: -A.T @ A

D_1d = np.ones((2, n_pix))
D_1d[0] *= -1
D_1d = sp.spdiags(D_1d, (0,1), n_pix, n_pix)
I = sp.identity(n_pix)
D = sp.vstack([
    sp.kron(D_1d, I),
    sp.kron(I, D_1d)
])

x, grad = SLQP_TV_min(
	f,
    grad_f,
    H_f,
    D,
    dim=n_pix**2,
    damp=1,
    max_iter=200,
    verbose=True,
    callback=lambda x, g: print(f(x))
)

plt.figure()
plt.imshow(x.reshape(n_pix, n_pix), cmap="gray")

plt.figure()
plt.imshow(grad.reshape(2*n_pix, n_pix), cmap="gray")

plt.show()