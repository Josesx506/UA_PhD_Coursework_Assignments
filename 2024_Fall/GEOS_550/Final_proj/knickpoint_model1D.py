import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# Define sumdiffs as a placeholder function
def sumdiffs(ac, bd, n=2):
    # Placeholder for a sumdiffs-like function
    diffs = np.diff(ac, n=n) + np.diff(bd, n=n) 
    return np.column_stack((ac[:-n], diffs)) # Return ac values and calculated diffs



# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8,3.8))
line1, = ax.plot([], [], linestyle="-", color="blue", linewidth=2, label="Elevation")
line2, = ax.plot([], [], color="black", linestyle="-", label="Sumdiffs")
text_knickpoint = ax.text(0, 0, "", fontsize=10, color="red")

# Initialize plot limits and labels
ax.set_ylim(2.8, -0.5)
ax.set_xlim(-1, 21)
ax.set_xlabel("distance [km]")
ax.set_ylabel("elevation [km]")
ax.set_title("Single Knickpoint Model")

# Initialization function for the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    text_knickpoint.set_text("")
    return line1, line2, text_knickpoint

# Update function for each frame
def update(i):
    a = np.arange(1, i+1)
    b = np.log10(a)
    k1 = 0.25-(i/1000)
    c = np.arange(i+1, 201)
    d = np.log10(c) + k1
    
    ac = np.concatenate((a, c))/10
    bd = np.concatenate((b, d))
    bd = bd[::-1]
    acbd = sumdiffs(ac, bd, 2)
    
    # Update the elevation line
    line1.set_data(ac, bd)
    
    # Update the sumdiffs line
    line2.set_data(acbd[:, 0], acbd[:, 1])
    
    # Update the knickpoint text
    if i > 0:
        text_knickpoint.set_position((ac[200-i]-1, bd[200-i] - 0.5))
        text_knickpoint.set_text("knickpoint")
    else:
        text_knickpoint.set_text("")
    
    return line1, line2, text_knickpoint

# Create the animation
ani = FuncAnimation(fig, update, frames=range(200, 0, -1), init_func=init, blit=False, interval=20)
ani.save("outfiles/knickpoint_animation.mp4", writer="ffmpeg", dpi=200)
