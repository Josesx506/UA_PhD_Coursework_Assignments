### Joses Omojola Submission

For my assignment, I tried to simulate the Wallace Creek morphology to evaluate how much of the channel would exist along the fault strike if deposition occured on the 
hanging wall. Using the `lem2dpackage3`, I retained the original topography, fault mask, and periodic boundary condition (BC) files. The BC mask uses Neumann (key=2) 
boundary conditions on the north and south edges of the grid, periodic bounds (key=3) on the east and west edges, and removes any conditions from internal grid cells (key=0). 
Using the original input parameters, the simulation got stuck at 772 kyr (for over 1 hour) and the step size decreased drammatically, so I stopped all simulations at 
750 kyr for the rest of this assignment. I did multiple tests to evaluate how fast deposition will occur on the hanging wall of the fault.

- In the first test, I set `U_h = 0.50 m/kyr`and `deposition velocity = 1e-3 m/kyr`. This simulation run was slow and I stopped it after an hour (Only 1 output scene was 
generated).
- In the second test, I reduced vertical uplift along the fault plane `U_v = 0.01 m/kyr` while keeping other parameters from test 1 constant. This test was also cancelled 
because of slow runtimes.
- Finally, I reduced `U_h = 0.35 m/kyr`, `U_v = 0.025 m/kyr`, and `deposition velocity = 1e-4  m/kyr`. This allowed formation of small alluvial fans near the mouth of each 
channel on the hanging wall early in the sumlation. The horizontal slip of the fault was too fast and deposited sediments were smeared along the fault plane, creating an 
inclined ramp at the end of the simulation. With deposition activated, the erosion of the footwall channels was limited and the eroded valleys were not quite as deep as 
the original simulation.

Increasing the deposition rate resulted in slower simulations, and I'm not sure if this is because the entrainment variable `K_0 = 1e-3 m^(1-p)/kyr` or whether the soil 
production was too little `P_0 = 0.1 m/kyr`. In summary, I wasn't able to directly replicate the Wallace Creek example because my horizontal fault slip was too fast but I 
learnt more about how structural processes affect landscape evolution.