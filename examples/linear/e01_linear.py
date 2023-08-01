import astra
import pylops
import numpy as np
from matplotlib import pyplot as plt
from tomopy.misc.phantom import shepp2d
from imsolve import BBLS, CGLS, SIRT

# create phantom
n_pix=512
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
x_SIRT = SIRT(A, b, verbose=True, max_iter=20)
x_BBLS = BBLS(A, b, verbose=True, max_iter=20)
x_CGLS = CGLS(A, b, verbose=True, max_iter=20)

plt.figure()
plt.title("SIRT, iteration 20")
plt.imshow(x_SIRT.reshape(n_pix, n_pix), cmap="gray")

plt.figure()
plt.title("BBLS, iteration 20")
plt.imshow(x_BBLS.reshape(n_pix, n_pix), cmap="gray")

plt.figure()
plt.title("CGLS, iteration 20")
plt.imshow(x_CGLS.reshape(n_pix, n_pix), cmap="gray")

plt.show()