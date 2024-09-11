import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


def create_fault_mask(nrows,ncols,amp=5,vert_offset=0.45,angle=20,plot=False,outpath="faultmask.txt"):
    """
    Create a fault mask for lem2D simulations that has a sine wave boundary

    Args:
        nrows (int): _description_
        ncols (int): number of columns in the model grid
        amp (int, optional): amplitude of sine wave fault boundary. Defaults to 5.
        vert_offset (int, optional): Shift the boudaries of fault mask along the vertical axis. Defaults to 0.45.
        angle (int, optional): Rotate the sine boundary. Defaults to 20.
        plot (bool, optional): Plot the mask. Defaults to False.
        outpath (str, optional): Path to save the mask. Defaults to "faultmask.txt".
    
    Returns:
        None: Saves the file to the specified path
    """
    # Define the size of the 2D array
    nrows, ncols = 101, 101

    # Create a 2D array of zeros
    grid = np.zeros((nrows, ncols))

    # Create the sine-wave boundary
    x = np.linspace(0, np.pi * 4, ncols)
    y = amp * np.sin(x)  # Adjust sine wave amplitude
    y = int(vert_offset * nrows) + y  # vertically shift the sine wave along the y-axis

    # partition grid to create binary mask(0 and 1)
    for col in range(ncols):
        wave_boundary = int(y[col])
        grid[:wave_boundary, col] = 1

    # Rotate the grid to create mimic a fault
    rotated_grid = rotate(grid, angle=angle, reshape=False, mode="reflect")
    
    # Plotting syntax
    if plot:
        plt.imshow(rotated_grid, aspect="auto")
        plt.show()

    # Flatten the array, and scale between 0-255
    flattened_grid = rotated_grid.flatten() * 255

    # Save as a text file
    np.savetxt(outpath, flattened_grid, fmt="%d")
    
    return None


if __name__ == "__main__":
    nrow = int(sys.argv[1])
    ncol = int(sys.argv[2])
    if len(sys.argv)==3:
        create_fault_mask(nrows=nrow,ncols=ncol)
    elif len(sys.argv)==3:
        path = sys.argv[3]
        create_fault_mask(nrows=nrow,ncols=ncol, outpath=path)
    