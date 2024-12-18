import matplotlib.pyplot as plt
import numpy as np
from devito import *
from examples.seismic import Model, plot_shotrecord, plot_velocity
from examples.seismic.source import Receiver, RickerSource, TimeAxis
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["animation.html"] = "jshtml"

from sympy import init_printing, latex

init_printing(use_latex="mathjax")

# Some ploting setup
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("xtick", labelsize=20) 
plt.rc("ytick", labelsize=20)

# ModelElastic
nx = 201      # grid points in x
nz = 201      # grid points in z
dx = 75.0     # grid increment in x
so = 8        # spatial order
to = 2        # time order
bl = 200      # boundary cells

# Define physical properties
vp = np.ones((nz, nx)) * 4.5
vs = np.ones((nz, nx)) * 2.8
rho = np.ones((nz, nx)) * 2.8 # Density

w0 = 150
rw0 = 0
for ii in range(25,65,1):
    vp[rw0,ii:ii+w0] = vp[rw0,ii:ii+w0]/1.5
    vs[rw0,ii:ii+w0] = vs[rw0,ii:ii+w0]/1.5
    rho[rw0,ii:ii+w0] = 2.3
    w0 -= 2
    rw0 += 1
w0 = 100
rw0 = 0
for ii in range(50,70,2):
    vp[rw0,ii:ii+w0] = vp[rw0,ii:ii+w0]/1.2
    vs[rw0,ii:ii+w0] = vs[rw0,ii:ii+w0]/1.2
    rho[rw0,ii:ii+w0] = 1.8
    w0 -= 4
    rw0 += 1

mu = rho * vs**2
lam = rho * vp**2 - 2 * mu

model = Model(origin=(0,0), vp=vp.T, vs=vs.T, b=1/rho.T, lam=lam.T, mu=mu.T, 
              shape=(nx, nz), spacing=(dx, dx), space_order=so, nbl=bl, bcs="mask")

aspect_ratio = model.shape[0]/model.shape[1]
slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]

# Define source properties and simulation duration
t0, tn = 0., 6000.   # Sim start-end time in ms
dt = model.critical_dt
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=model.grid, f0=0.004, time_range=time_range)
src.coordinates.data[:] = [7500., 7500.]

# src.show()

# Now we create the velocity and pressure fields
x, z = model.grid.dimensions
t = model.grid.stepping_dim
time = model.grid.time_dim
s = time.spacing

v = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=to, save=time_range.num)
tau = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=to)

# The source injection term
src_xx = src.inject(field=tau.forward[0, 0], expr=s*src)
src_zz = src.inject(field=tau.forward[1, 1], expr=s*src)

rz = 5 # Depth of receivers

# The receiver terms for the stress and velocity amplitudes
nrec = 201
rec = Receiver(name="rec", grid=model.grid, npoint=nrec, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec.coordinates.data[:, -1] = rz

rec2 = Receiver(name="rec2", grid=model.grid, npoint=nrec, time_range=time_range)
rec2.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec2.coordinates.data[:, -1] = rz

rec3 = Receiver(name="rec3", grid=model.grid, npoint=nrec, time_range=time_range)
rec3.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec3.coordinates.data[:, -1] = rz

rec_term = rec.interpolate(expr=tau[0, 0] + tau[1, 1])
rec_term += rec2.interpolate(expr=v[1])
rec_term += rec3.interpolate(expr=v[0])


# Plot decimated receiver geometries
plt_options_model = {"cmap": "jet", "extent": [model.origin[0], model.origin[0] + model.domain_size[0],
                                               model.origin[1] + model.domain_size[1], model.origin[1]]}

fig,ax = plt.subplots(1,3,figsize=(16,3),sharey=True)
ax[0].imshow(lam, aspect="auto", **plt_options_model),ax[0].set_title(r"$\tau_{xx} + \tau_{zz}$")
ax[0].scatter(rec.coordinates.data[::20, 0],rec.coordinates.data[::20, 1],s=25, c="green", marker="D")
ax[1].imshow(vp, aspect="auto", **plt_options_model),ax[1].set_title(r"$V_{y}$")
ax[1].scatter(rec2.coordinates.data[::20, 0],rec2.coordinates.data[::20, 1],s=25, c="green", marker="D")
ax[2].imshow(vp, aspect="auto", **plt_options_model),ax[2].set_title(r"$V_{x}$")
ax[2].scatter(rec3.coordinates.data[::20, 0],rec3.coordinates.data[::20, 1],s=25, c="green", marker="D")

plt.show()

# Create the staggered updates
l, mu, ro = model.lam, model.mu, model.b    # Lame parameters

# First order elastic wave equation
pde_v = v.dt - ro * div(tau)
pde_tau = tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) + grad(v.forward).transpose(inner=False))
# Time update
u_v = Eq(v.forward, model.damp * solve(pde_v, v.forward))
u_t = Eq(tau.forward,  model.damp * solve(pde_tau, tau.forward))

op = Operator([u_v] + [u_t] + src_xx + src_zz + rec_term)
# Full run for 6 secs to plot the wavefield
dtm = model.critical_dt
op(dt=dtm, time=int(tn/dtm))

# Create the wavefield animation snapshots
plt.ioff()
scale = .5*1e-3
divi = 1
plt_options_model = {"extent": [model.origin[0] , model.origin[0] + model.domain_size[0],
                                model.origin[1] + model.domain_size[1], model.origin[1]],"aspect":"auto"}
# Initialize animated plot
fig,ax = plt.subplots(figsize=(8,5))
image = ax.imshow(np.transpose(v[1].data[0][slices]), vmin=-scale, vmax=scale, **plt_options_model, cmap="seismic")
ax.imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, alpha=.5, **plt_options_model, cmap="jet")
ax.set_title(r"$V_{y}$"), ax.set_xlabel("Distance (m)"), ax.set_ylabel("Depth (m)")

# Animate the pressure fields
def update(itr):
    image.set_array(np.transpose(v[1].data[itr*divi][slices]))
    return [image]

ani = FuncAnimation(fig, update, frames=int(v[1].data.shape[0]/divi), blit=False, repeat=False)
ani.save(filename="output/elastic_2D.mp4", writer="ffmpeg", fps=30, dpi=200, savefig_kwargs={"pad_inches": 0})


# Plot the receiver recordings
plt_options_model = {"cmap": "gray", "extent": [model.origin[0], model.origin[0] + model.domain_size[0]/1e3, tn/1e3, t0]}
r_sum = rec.data + rec2.data + rec3.data

# Calculate the scales of the data
rc1_sc,rc2_sc = np.max(rec.data) / 10.,np.max(rec2.data) / 10.
rc3_sc,rc4_sc = np.max(rec3.data) / 2.,np.max(r_sum) / 20.

fig,ax = plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
ax = ax.flatten()
ax[0].imshow(rec.data, aspect="auto", vmin=-rc1_sc, vmax = rc1_sc, **plt_options_model),ax[0].set_title(r"$\tau_{xx} + \tau_{zz}$")
ax[1].imshow(rec2.data, aspect="auto", vmin=-rc2_sc, vmax = rc2_sc, **plt_options_model),ax[1].set_title(r"$V_{y}$")
ax[2].imshow(rec3.data, aspect="auto", vmin=-rc3_sc, vmax = rc3_sc, **plt_options_model),ax[2].set_title(r"$V_{x}$")
ax[3].imshow(r_sum, aspect="auto", vmin=-rc4_sc, vmax = rc4_sc, **plt_options_model),ax[3].set_title("Sum Receivers")
ax[0].set_ylabel("Time (s)"),ax[2].set_ylabel("Time (s)"),ax[2].set_xlabel("Distance (km)"),ax[3].set_xlabel("Distance (km)")

plt.savefig("output/elastic_rcv_2D.png",bbox_inches="tight",dpi=200)
plt.close()