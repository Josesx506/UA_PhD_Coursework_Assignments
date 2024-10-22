import numpy as np
import xarray as xr

dem = xr.open_dataarray("lds-nz-8m-DEM12-GTiff/dem_100x100.nc")

def create_uplift_mask(dem,cutoff=150):
    mask = dem.data
    mask = np.where(mask<cutoff,0,1)
    # Flatten the array, and scale between 0-255
    fmask = mask.flatten() * 255
    # Save as a text file
    np.savetxt("faultmask.txt", fmask, fmt="%d")
    return mask

def create_input_elv(dem,div_pct=1.1):
    """
    Scale the input elevation to be several order of magnitudes smaller
    than the original values

    Args:
        dem (xr.DataArray): DEM data from netcdf file
        div_pct (float, optional): Scaling magntude. Defaults to 1.1.

    Returns:
        np.array: scaled elevation array
    """
    elv = dem.data
    max_val = elv.max()*div_pct
    elv = elv/max_val
    # Flatten the array
    felv = elv.flatten()
    # Save as a text file
    np.savetxt("inputtopo.txt", felv, fmt="%f")
    return elv

def create_boundary_conditions(dem,bcv=[2,2,2,2]):
    """
    Boundary conditions mask

    Args:
        dem (xr.DataArray): DEM data from netcdf file
        bcv (list, optional): Options for boundary conditions arranged as [left,top,right,bottom]. Defaults to [2,2,2,2].

    Returns:
        np.array: scaled elevation array
    """
    data = dem.data
    bc_mask = np.zeros_like(data)
    bc_mask[:,0] = bcv[0]
    bc_mask[0,:] = bcv[1]
    bc_mask[:,-1] = bcv[2]
    bc_mask[-1,:] = bcv[3]
    # Flatten the array
    fbc = bc_mask.flatten()
    # Save as a text file
    np.savetxt("BCmask.txt", fbc, fmt="%d")
    return bc_mask

mask = create_uplift_mask(dem,100)
elv = create_input_elv(dem)
bc = create_boundary_conditions(dem, [2,2,2,2])

# Plot channel elevation along strike

def euclidean_distance(pf):
    # Calculate Euclidean distances along the points in p1, starting with 0 for the first point
    distances = np.sqrt(np.diff(pf["x"])**2 + np.diff(pf["y"])**2)
    return np.insert(np.cumsum(distances), 0, 0)


# fig = plt.figure(figsize=(9,4))
# for prf in profiles:
#     x = xr.DataArray(prf["x"], dims="z")
#     y = xr.DataArray(prf["y"], dims="z")
#     el = dem.interp(x=x,y=y,method="linear")
#     dist = euclidean_distance(prf)