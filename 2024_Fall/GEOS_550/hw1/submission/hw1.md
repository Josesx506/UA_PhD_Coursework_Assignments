### Joses Omojola - HW1
Use lem2d.c to simulate the motion of a fault by modifying the parameters U_v, U_h, and horizontalslipvector in the input file “input.txt” and (if needed) modifying faultmask.txt (which specifies the grid points that are part of the displacing block and the grid points that are stationary). You can choose to model a simple linear fault. Or, you can simulate a real fault using some data from a fault map. Your writeup should include a short description of what you changed in the input files or source code, an animation of your fault in motion, and a short description of whether the outcome was what you expected or were aiming for. If you have trouble, I am happy to help troubleshoot. <br>

### Response
For this assignment, I modified the fault mask to use a simplified fault with a sine wave geometry running from NW-SE. 
- In the `input.txt` file, U_v was updated to 0.25 m/kyr,  U_h was updated to 2 m/kyr, and the horizontalslipvector was set to 90˚. 
    - The simulation was run for 300 kyr, and the max topography height expected was 75 m which aligns with the color bar in the animation. 
    - The zeroed segment of the fault mask remained stagnant, raising the elevation of footwall throughout the simulation
    - Sometimes when the `horizontalslipvector` was not directly aligned with the fault mask, the simulation got stuck with negative elevations.
- The xyz output files were converted to tiff using gdal and a bash script. 
- In Paraview, the `topo.tif*` image stack was imported, and the `Warp by Scalar` filter was applied. The color bar was adjusted to "Update on Apply", and the final output was exported as an avi animation.