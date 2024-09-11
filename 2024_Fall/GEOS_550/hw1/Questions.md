### HW1

Use lem2d.c to simulate the motion of a fault by modifying the parameters U_v, U_h, and horizontalslipvector in the input file “input.txt” and (if needed) modifying faultmask.txt (which specifies the grid points that are part of the displacing block and the grid points that are stationary). You can choose to model a simple linear fault. Or, you can simulate a real fault using some data from a fault map. Your writeup should include a short description of what you changed in the input files or source code, an animation of your fault in motion, and a short description of whether the outcome was what you expected or were aiming for. If you have trouble, I am happy to help troubleshoot. <br>

Reminder: Using ***gcc***, the source code should be compiled using `gcc -g -o lem2d lem2d.c util.c userdefinedfunctions.c`. <br>

Configuration can be defined in the `lem2dpackage/input.txt`. The current grid dimensions are 101 by 101 cells, where each cell has a node spacing of 30 m, creating a *3 by 3 km* model grid. The 
fault mask can be created using the pyton script `python create_mask.py 101 101`, and matching the 
input rows and columns.
