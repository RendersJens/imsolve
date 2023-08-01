import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from imsolve import barzilai_borwein
from imsolve.experimental import lagrange_newton_BB

# create phantom
n_pix = 256
phantom = shepp2d(n_pix)[0]/255

# creat CT projection operator
vol_geom = astra.create_vol_geom(*phantom.shape)
angles = np.linspace(0,np.pi,300)
proj_geom = astra.create_proj_geom('parallel', 1, n_pix, angles)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
A = astra.optomo.OpTomo(proj_id)
A = 1/n_pix * pylops.VStack([A])

# simulate CT scan
I_0 = 1e5
p = I_0 * np.exp(-A @ phantom.ravel())
p = np.random.poisson(p)/I_0

# log normalize
b = -np.log(p)

# solve
f = lambda x: 1/2*np.linalg.norm(A @ x - b)**2
grad_f = lambda x: A.T @ (A @ x - b)
H_f = lambda x: A.T @ A

grad_g = lambda x: -A.T @ (np.exp(-A @ x) - p)
H_g = lambda x: A.T @ pylops.Diagonal(np.exp(-A @ x)) @ A

#x = barzilai_borwein(grad_f, dim=n_pix**2, max_iter=20, verbose=True)
x_LNBB = lagrange_newton_BB(f, grad_f, H_f, bounds=(0, np.inf), dim=n_pix**2, outer_iter=200, inner_iter=30, verbose=True)
x_BB = barzilai_borwein(grad_f, bounds=(0, np.inf), dim=n_pix**2, max_iter=20, verbose=True)

plt.figure()
plt.imshow(x_LNBB.reshape(n_pix, n_pix), cmap="gray")

plt.figure()
plt.imshow(x_BB.reshape(n_pix, n_pix), cmap="gray")

plt.show()