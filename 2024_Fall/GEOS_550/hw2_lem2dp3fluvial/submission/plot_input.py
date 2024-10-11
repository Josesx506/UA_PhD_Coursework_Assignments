import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from scipy.ndimage import rotate

gx,gy = 101,101
bc = np.loadtxt("../BCmask.txt")
bc = rotate(bc.reshape(gx,gy),0)

fm = np.loadtxt("../faultmask.txt")
fm = rotate(fm.reshape(gx,gy), 0)

topo = np.loadtxt("../inputtopo.txt")
topo = rotate(topo.reshape(gx,gy),0)

fig, ax = plt.subplots(2,2, figsize=(10,9), gridspec_kw={"wspace":0.3},sharex=True,sharey=True)
ax = ax.flatten()
cmap1 = cm["viridis"].resampled(4)
cmap2 = cm["viridis"].resampled(2)
im1 = ax[0].imshow(bc,aspect="auto",cmap=cmap1)
im2 = ax[1].imshow(fm,aspect="auto",cmap=cmap2)
im3 = ax[2].imshow(topo,cmap="gist_earth")

cax1 = fig.add_axes([0.48, 0.53, 0.01, 0.34])
cbar = fig.colorbar(im1, cax=cax1,format="%1i")
cbar.ax.locator_params(nbins=4)
cax2 = fig.add_axes([0.92, 0.53, 0.01, 0.34])
cbar = fig.colorbar(im2, cax=cax2,format="%1i")
cbar.ax.locator_params(nbins=1)
cax3 = fig.add_axes([0.48, 0.11, 0.01, 0.34])
cbar = fig.colorbar(im3, cax=cax3,)

ax[0].set_title("Boundary Condition mask")
ax[1].set_title("Fault mask")
ax[2].set_title("Initial Topography (m)")
ax[3].axis("off")

plt.savefig("Input_maps.png",bbox_inches="tight",dpi=200)
plt.close()