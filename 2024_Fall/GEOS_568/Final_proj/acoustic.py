import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["animation.html"] = "jshtml"

# Simple finite difference solver 
# Acoustic wave equation  p_tt = c^2 p_xx + src
# 2-D regular grid

nx = 200      # grid points in x
nz = 200      # grid points in z
nt = 1000      # number of time steps
dx = 75.0     # grid increment in x
dt = 0.0075    # Time step
c0 = 3500.0   # velocity (can be an array)
isx = nx // 2  # source index x
isz = nz // 2  # source index z
ist = 100     # shifting of source time function
f0 = 15.0    # dominant frequency of source (Hz)
isnap = 10     # snapshot frequency
T = 1.0 / f0  # dominant period
nop = 5       # length of operator
sIdx = 0     # Start model index
if nop == 3:
    dw = 1
elif nop == 5:
    dw = 2


# Model type, available are "homogeneous", "fault_zone",
# "surface_low_velocity_zone", "random", "topography",
# "slab"
model_type = "basin"

# Receiver locations
irx = np.array([sIdx+18, sIdx+80, sIdx+110, sIdx+150, sIdx+178])
irz = np.array([7, 7, 7, 7, 7])
seis = np.zeros((len(irx), nt))

# Initialize pressure at different time steps and the second
# derivatives in each direction
p = np.zeros((nz, nx))
pold = np.zeros((nz, nx))
pnew = np.zeros((nz, nx))
pxx = np.zeros((nz, nx))
pzz = np.zeros((nz, nx))

# Initialize velocity model
c = np.zeros((nz, nx))


if model_type == "fault_zone":
    c += c0
    c[:, nx // 2 - 5: nx // 2 + 5] *= 0.8    
elif model_type == "topography":
    c += c0
    c[0 : 10, 10 : 50] = 0                         
    c[0 : 10, 105 : 115] = 0                       
    c[0 : 30, 145 : 170] = 0
    c[10 : 40, 20 : 40]  = 0
    c[0 : 15, 50 : 105] *= 0.8    
elif model_type == "slab":
    c += c0
    c[110 : 125, 0 : 125] = 1.4 * c0
    for i in range(110, 180):
        c[i , i-5 : i + 15 ] = 1.4 * c0
elif model_type == "basin":
    c += c0
    w0 = 150
    rw0 = 0
    for ii in range(sIdx+25,sIdx+65,1):
        c[rw0,ii:ii+w0] = c[rw0,ii:ii+w0]/1.5
        w0 -= 2
        rw0 += 1
    w0 = 100
    rw0 = 0
    for ii in range(sIdx+50,sIdx+70,2):
        c[rw0,ii:ii+w0] = c[rw0,ii:ii+w0]/1.2
        w0 -= 4
        rw0 += 1
else:
    raise NotImplementedError
    
cmax = c.max()

# Source time function Gaussian, nt + 1 as we loose the last one by diff
src = np.empty(nt + 1)
for it in range(nt):
    src[it] = np.exp(-1.0 / T ** 2 * ((it - ist) * dt) ** 2)
# Take the first derivative
src = np.diff(src) / dt
src[nt - 1] = 0

v = max([np.abs(src.min()), np.abs(src.max())])

plt.ioff()
# Initialize animated plot
fig,ax = plt.subplots(figsize=(5,5))
image = ax.imshow(pnew[sIdx:sIdx+200,:200], interpolation="nearest",
                   vmin=-v, vmax=+v, cmap=plt.cm.RdBu,extent=[sIdx,sIdx+200,200,0])
# Plot the receivers
for x, z in zip(irx, irz):
    ax.text(x, z, "+")
ax.text(isx, isz, "o")
# ax.colorbar()
ax.set_xlabel("ix")
ax.set_ylabel("iz")


# required for seismograms
ir = np.arange(len(irx))

# Output Courant criterion
alpha = cmax*dt/dx
alpha2 = alpha**2
print(f"Courant Criterion eps : {alpha:.2f}")
kappa = (1 - alpha) / (1 + alpha)

pres_hist = []
pmax_hist = []


# Time extrapolation
for it in range(nt):
    if nop==3:
        # calculate partial derivatives, be careful around the boundaries
        for i in range(dw, nx - dw):
            pzz[:, i] = p[:, i + 1] - 2 * p[:, i] + p[:, i - 1]
        for j in range(dw, nz - dw):
            pxx[j, :] = p[j - 1, :] - 2 * p[j, :] + p[j + 1, :]

    if nop==5:
        # calculate partial derivatives, be careful around the boundaries
        for i in range(dw, nx - dw):
            pzz[:, i] = -1./12*p[:,i+2]+4./3*p[:,i+1]-5./2*p[:,i]+4./3*p[:,i-1]-1./12*p[:,i-2]
        for j in range(dw, nz - dw):
            pxx[j, :] = -1./12*p[j+2,:]+4./3*p[j+1,:]-5./2*p[j,:]+4./3*p[j-1,:]-1./12*p[j-2,:]
                    
            
    pxx /= dx ** 2
    pzz /= dx ** 2

    # Time extrapolation
    pnew = 2 * p - pold + (dt ** 2) * (c ** 2) * (pxx + pzz)
    # Add source term at isx, isz
    pnew[isz, isx] = pnew[isz, isx] + src[it]

    # Apply absorbing boundary conditions to the left,right, and bottom
    if nop==3:
        pnew[1:nz-1,0] = p[1:nz-1,1] - (kappa * (pnew[1:nz-1,1] - p[1:nz-1,0]))        # Left boundary
        pnew[1:nz-1,-1] = p[1:nz-1,-2] - (kappa * (pnew[1:nz-1,-2] - p[1:nz-1,-1]))    # Right boundary
        pnew[-1,1:nx-1] = p[-2,1:nx-1] - (kappa * (pnew[-2,1:nx-1] - p[-1,1:nx-1]))    # Bottom boundary
    if nop==5:
        pnew[dw:nz-dw,:2] = p[dw:nz-dw,1:3] - (kappa * (pnew[dw:nz-dw,1:3] - p[dw:nz-dw,:2]))          # Left boundary
        pnew[dw:nz-dw,-2:] = p[dw:nz-dw,-3:-1] - (kappa * (pnew[dw:nz-dw,-3:-1] - p[dw:nz-dw,-2:]))    # Right boundary
        pnew[-2:,dw:nx-dw] = p[-3:-1,dw:nx-dw] - (kappa * (pnew[-3:-1,dw:nx-dw] - p[-2:,dw:nx-dw]))    # Bottom boundary

    if it % isnap == 0:    # you can change the speed of the plot by increasing the plotting interval
        pres_hist.append(pnew)
        pmax_hist.append(p.max())

    pold, p = p, pnew

    # Save seismograms
    seis[ir, it] = p[irz[ir], irx[ir]]


if it % isnap == 0:    # you can change the speed of the plot by increasing the plotting interval
    pres_hist.append(pnew)
    pmax_hist.append(p.max())

# Animate the pressure fields
def update(itr):
    # ax.clear()
    ax.set_title(f"Time {itr*isnap*dt:.2f} | Max P: {pmax_hist[itr]:.2f}")
    image.set_array(pres_hist[itr][sIdx:sIdx+200,:200])
    return [image]

ani = FuncAnimation(fig, update, frames=len(pres_hist), blit=False, repeat=False)
ani.save(filename="output/acoustic_2D.mp4", writer="ffmpeg", fps=5, dpi=200, savefig_kwargs={"pad_inches": 0})

# Plot the source time function and the seismograms 
for i in range(500): plt.close()

plt.ioff()
# plt.figure(figsize=(12, 12))
gridspec = dict(hspace=0.3, height_ratios=[0.2, 1, 1])
fig, ax = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw=gridspec)

s_ar_t = [3.25,3,2.95,3.1,3.2] # S-arrival times
ss_ar_t = [3.25,3.65,3.61,3.73,3.2]

time = np.arange(nt) * dt
ax[0].plot(time, src)
ax[0].set_title("Source time function")
ax[0].set_xlabel("Time (s) ")
ax[0].set_ylabel("ampl.")
ax[0].margins(x=0.01)

# Plot the receivers and source. The velocity model is influenced by the Earth model above
ax[1].set_title("Velocity Model")
for ir,(ir_x,ir_z) in enumerate(zip(irx,irz)):
    if ir in [1,2,3]: cl="w" 
    else: cl="k"
    ax[1].text(ir_x*dx-75,ir_z*dx+750,s=ir+1,c=cl,zorder=2)
ax[1].scatter(irx*dx, irz*dx, marker="v",ec="k",c="cyan",zorder=2)
ax[1].scatter(isx*dx, isz*dx, marker="*", ec="k", c="w",zorder=2, s=250)
img = ax[1].imshow(c, aspect="auto", extent=[0,nx*dx,nz*dx,0])
ax[1].set_xlabel("Distance (m)")
ax[1].set_ylabel("Depth (m)")
div = make_axes_locatable(ax[1])
cax = div.append_axes("right", "2%", pad="3%")
cbar = fig.colorbar(img, cax=cax, shrink=0.6)
cbar.ax.set_ylabel("Velocity (m/s)")

ymax = seis.ravel().max()
for ir in range(len(seis)):
    ax[2].plot(time, (seis[ir, :]/ymax) + ir+1)
    ax[2].scatter(ss_ar_t[ir],ir+1,c="b",marker="|",lw=1,s=200,zorder=2)
    ax[2].scatter(s_ar_t[ir],ir+1,c="k",marker="|",lw=1,s=200,zorder=2)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Station Number")
ax[2].margins(x=0.01)

ax[0].text(-0.8,20,"(a)")
ax[1].text(-1500,-750,"(b)")

plt.savefig("output/acoustic_2D.png",bbox_inches="tight",dpi=200)
plt.close()


# Calculate the basin depth
for ii,ir in enumerate(irx):
    cmn = c[:,ir].mean()               # mean 1D velocity
    rcvdp = (irz[ii]+1) * dx           # receiver depth in model
    stidf = ss_ar_t[ii] - s_ar_t[ii]   # s-ss time difference
    bsdp = int((stidf/(1/cmn)))           # Estimated basin depth
    tbsidx = np.where(c[:,ir] == 3500)[0][0]
    tbsdp = int(tbsidx * dx)              # True basin depth
    print(bsdp, tbsdp)