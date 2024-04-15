import matplotlib.pyplot as plt
import numpy as np

def geotherm_gradient(t0,q0,k,d,z0,A):
    z = np.arange(z0,-d,-1000)

    fig,ax = plt.subplots(figsize=(7,4))
    colors = ["C0","r","g"]
    for i,h in enumerate(A):
        temp = temperature(t0,q0,k,z,h)
        ax.plot(temp,z/1000,c=colors[i])
    plt.show()


def temperature(t0,q0,k,z,heat):
    T = (-(heat*(z**2))/(2*k)) + ((q0*z)/k) + t0
    return T

k = 2.5 # thermal conductivity
A=[1.25e-6, 2.5e-6] # radioactive heat generation
q0 = 0.021 # heat flow
t0 = 0 # surface temperature
z0 = 0
d = 50000

geotherm_gradient(t0,q0,k,d,z0,A)