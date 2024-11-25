import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import (arccos, arctan, asarray, cos, diff, linspace, outer, pi,
                   savetxt, sign, sin, sqrt, vectorize)

save_dir = "output"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def double_coup_rad_ptn(theta, phi):
    # FP (far-field P-wave) pattern
    A_FP_r = np.sin(2*theta) * np.cos(phi)
    
    # FS (far-field S-wave) pattern
    A_FS_theta = np.cos(2*theta) * np.cos(phi)
    A_FS_phi = -np.cos(theta) * np.sin(phi)
    
    # Total S-wave amplitude
    A_FS = A_FS_theta + A_FS_phi
    
    return A_FP_r, A_FS

def single_coup_rad_ptn(theta):

    # A_FP = np.cos(theta)
    # A_FS = np.sin(theta)
    
    A_FP = np.sin(2*theta)
    A_FS = 2 * (np.cos(theta)**2)

    # A_FP = 1 + (2 * (np.cos(theta)**2))
    # A_FS = 2 * np.sin(theta) * np.cos(theta)
    
    return A_FP, A_FS

# Create figure
with plt.rc_context({"font.family": "Times New Roman","font.size":12}):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6), gridspec_kw={"wspace":0.3}, subplot_kw={"projection": "polar"})

    for ax in axes:
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)
        # ax.set_theta_zero_location("N")

    # x1-x3 plane (phi = 0)
    theta_x1x3 = np.linspace(0, 2*np.pi, 360)
    phi_x1x3 = np.zeros_like(theta_x1x3)
    A_FP_x1x3, A_FS_x1x3 = double_coup_rad_ptn(theta_x1x3, phi_x1x3)

    # x1-x2 plane (theta = pi/2)
    phi_x1x2 = np.linspace(0, 2*np.pi, 360)
    theta_x1x2 = np.ones_like(phi_x1x2) * np.pi/2
    A_FP_x1x2, A_FS_x1x2 = double_coup_rad_ptn(theta_x1x2, phi_x1x2)

    # Plot x1-x3 plane
    axes[0].plot(theta_x1x3, np.abs(A_FP_x1x3), "r-", label="P-wave")
    axes[0].plot(theta_x1x3, np.abs(A_FS_x1x3), "b-", label="S-wave")
    axes[0].set_title(r"$x_1-x_3$ plane ($\phi = 0$)")
    axes[0].grid(True),axes[0].set_yticklabels([]),axes[0].legend(handlelength=0.5)
    axes[0].annotate("",  xy=(45*pi/180,0.15),  xytext=(315*pi/180, 0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)
    axes[0].annotate("",  xy=(225*pi/180, 0.15),  xytext=(135*pi/180,0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)

    # Plot x1-x2 plane
    axes[1].plot(phi_x1x2, np.abs(A_FP_x1x2), "r-", label="P-wave")
    axes[1].plot(phi_x1x2, np.abs(A_FS_x1x2), "b-", label="S-wave")
    axes[1].set_title(r"$x_1-x_2$ plane ($\theta = \frac{\pi}{2}$)")
    axes[1].grid(True),axes[1].set_yticklabels([]),axes[1].legend(handlelength=0.5)
    axes[1].annotate("",  xy=(45*pi/180,0.15),  xytext=(315*pi/180, 0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)
    axes[1].annotate("",  xy=(225*pi/180, 0.15),  xytext=(135*pi/180,0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)

    # Add figure title
    fig.suptitle("Double-Couple Source Radiation Patterns", y=0.98, size=14)
    plt.savefig(f"{save_dir}/double-couple_rad_ptn.png",dpi=200)
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6), gridspec_kw={"wspace":0.3}, subplot_kw={"projection": "polar"})

    for ax in axes:
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)

    # x1-x3 plane (phi = 0)
    theta_x1x3 = np.linspace(0, 2*np.pi, 360)
    phi_x1x3 = np.zeros_like(theta_x1x3)
    A_FP_x1x3, A_FS_x1x3 = single_coup_rad_ptn(theta_x1x3)

    # x1-x2 plane (theta = pi/2)
    phi_x1x2 = np.linspace(0, 2*np.pi, 360)
    theta_x1x2 = np.ones_like(phi_x1x2) * np.pi/2
    A_FP_x1x2, A_FS_x1x2 = single_coup_rad_ptn(theta_x1x2)

    # Plot x1-x3 plane
    axes[0].plot(theta_x1x3, np.abs(A_FP_x1x3), "r-", label="P-wave")
    axes[0].plot(theta_x1x3, np.abs(A_FS_x1x3), "b-", label="S-wave")
    axes[0].set_title(r"$x_1-x_3$ plane ($\phi = 0$)")
    axes[0].grid(True),axes[0].set_yticklabels([]),axes[0].legend(handlelength=0.5)
    axes[0].annotate("",  xy=(45*pi/180,0.15),  xytext=(315*pi/180, 0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)
    axes[0].annotate("",  xy=(225*pi/180, 0.15),  xytext=(135*pi/180,0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)

    # Plot x1-x2 plane
    axes[1].plot(phi_x1x2, np.abs(A_FP_x1x2), "r-", label="P-wave")
    axes[1].plot(phi_x1x2, np.abs(A_FS_x1x2), "b-", label="S-wave")
    axes[1].set_title(r"$x_1-x_2$ plane ($\theta = \frac{\pi}{2}$)")
    axes[1].grid(True),axes[1].set_yticklabels([]),axes[1].legend(handlelength=0.5)
    axes[1].annotate("",  xy=(45*pi/180,0.15),  xytext=(315*pi/180, 0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)
    axes[1].annotate("",  xy=(225*pi/180, 0.15),  xytext=(135*pi/180,0.15),  arrowprops=dict(facecolor="w", edgecolor="k", width=4, headwidth=8, lw=2),)

    # Add figure title
    fig.suptitle("Single-Couple Source Radiation Patterns", y=0.98, size=14)
    plt.savefig(f"{save_dir}/single-couple_rad_ptn.png",dpi=200)
    plt.close()


# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# phi = np.linspace(0, 2 * np.pi, 360)  # Azimuth angle
# theta = np.linspace(0, np.pi, 180)   # Polar angle

# # Radiation pattern functions
# def p_wave_radiation(theta, phi):
#     return np.sin(2 * theta) * np.cos(phi)

# def s_wave_radiation(theta, phi):
#     a_theta = np.cos(2 * theta) * np.cos(phi)  # Theta component
#     a_phi = -np.cos(theta) * np.sin(phi)       # Phi component
#     return np.sqrt(a_theta**2 + a_phi**2)      # Resultant amplitude

# # Evaluate patterns for x1-x3 plane (phi = 0)
# phi_fixed = 0
# p_x1_x3 = p_wave_radiation(theta, phi_fixed)
# s_x1_x3 = s_wave_radiation(theta, phi_fixed)

# # Evaluate patterns for x1-x2 plane (theta = pi/2)
# theta_fixed = np.pi / 2
# p_x1_x2 = p_wave_radiation(theta_fixed, phi)
# s_x1_x2 = s_wave_radiation(theta_fixed, phi)

# # Plotting
# fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 10))

# # P-wave in x1-x3 plane
# axes[0, 0].plot(theta, p_x1_x3, label='P-wave')
# axes[0, 0].set_title("P-wave in x1-x3 plane")
# axes[0, 0].legend()

# # S-wave in x1-x3 plane
# axes[0, 1].plot(theta, s_x1_x3, label='S-wave', color='orange')
# axes[0, 1].set_title("S-wave in x1-x3 plane")
# axes[0, 1].legend()

# # P-wave in x1-x2 plane
# axes[1, 0].plot(phi, p_x1_x2, label='P-wave')
# axes[1, 0].set_title("P-wave in x1-x2 plane")
# axes[1, 0].legend()

# # S-wave in x1-x2 plane
# axes[1, 1].plot(phi, s_x1_x2, label='S-wave', color='orange')
# axes[1, 1].set_title("S-wave in x1-x2 plane")
# axes[1, 1].legend()

# plt.tight_layout()
# plt.show()
