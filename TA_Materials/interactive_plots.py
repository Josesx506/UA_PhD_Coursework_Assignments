# Interactive plots for a jupyter notebook. Useful for explaining concepts as TA
# Will be modified in the future. Source: https://colab.research.google.com/drive/13ZkAcpoweTpbSXzGOPp61hGOau5iXfgu?usp=sharing#scrollTo=aX9VJ_NyizzO

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact


@interact(GradPp=(0.4,0.9),GradSv=(0.9,1.3),zw=(0,5000,100)) # Widget variables: GradPp [psi/ft], GradSv [psi/ft], z_w [ft]
def plotter(GradPp=0.5,GradSv=1,zw=0):
    # define linear space for depth z
    z = np.linspace(zw,10000,100) # [ft]
    # calculate pore pressure and vertical stress
    rhowg = 0.44 # [psi/ft]
    Pp = rhowg*zw + GradPp*(z-zw)
    Sv = rhowg*zw + GradSv*(z-zw)
    # plot pore pressure, vertical stress and seafloor
    plt.plot(Pp,z, 'b-', label = "P_p")
    plt.plot(Sv,z, 'r-', label = "S_v")
    plt.plot([0,15000],[zw,zw],'k--', label = "Seafloor")
    # plotting options
    plt.xlabel('Pore pressure and stress [psi]')
    plt.ylabel('Depth [ft]')
    plt.xlim(0,15000)
    plt.ylim(0,10000)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()