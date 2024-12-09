import argparse

import numpy as np
import xarray as xr
from landlab import RasterModelGrid
from landlab.components import DepressionFinderAndRouter, FlowAccumulator
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, gaussian_filter


def interpolate_mask(mask, edge_values):
    """
    Interpolate values in a 2D mask using edge values.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        2D array of zeros and ones representing the channel mask
    edge_values : numpy.ndarray
        2D array of elevation values around the edges of the mask
    
    Returns:
    --------
    numpy.ndarray
        Interpolated elevation map with the same shape as the input mask
    """
    # Find coordinates of edge points
    edge_coords = np.argwhere(edge_values != 0)
    edge_z_values = edge_values[edge_coords[:, 0], edge_coords[:, 1]]
    
    # Find coordinates of points to be interpolated (where mask is 1)
    interpolation_coords = np.argwhere((mask == 1) & (edge_values == 0))
    
    # Create a grid of all points
    grid_x, grid_y = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    
    # Perform interpolation
    interpolated_values = griddata(
        points=edge_coords, 
        values=edge_z_values, 
        xi=(interpolation_coords[:, 0], interpolation_coords[:, 1]), 
        method="linear"
    )
    
    # Create output array
    output = np.zeros_like(mask, dtype=float)
    output[edge_coords[:, 0], edge_coords[:, 1]] = edge_z_values
    output[interpolation_coords[:, 0], interpolation_coords[:, 1]] = interpolated_values
    output[np.isnan(output)] = 0
    
    return output


def smooth_dem_drainage(input_file:str,
                        out_file:str,
                        cell_size: int = 100,
                        min_chn_sz:float = 1e6,
                        sigma:int = 8,
                        mask_buf:int = 5,
                        elv_diff:int = 80
                        ):
    """
    Given a DEM map, estimate the drainage area. Smooth the topography only within 
    the drainage basins (channels) and drop the elevation by a limited height.
    This is useful for providing input conditions for knickpoints where the smoothed 
    channels serve as pre-erosion topography. The elevation is also used to ensure 
    we don"t have to deal with uplifting topography from 0 during the landscape 
    evolution modeling process.

    Args:
        input_file (str): input file path
        out_file (str): output file path
        cell_size (int, optional): cell size in m. Defaults to 100.
        min_chn_sz (float, optional): Minimum channel size (volume being drained into channel). Defaults to 1e6.
        mask_buf (int, optional): Buffer the channel mask by n- cells in all directions. Defaults to 7.
        elv_diff (int, optional): Elevation drop in m, to allow the LEM to run uplift simulations. Defaults to 10
    """
    # Read file in and estimate the number of raster rows & columns
    dem = xr.open_dataarray(input_file)
    tmp = dem.copy()
    nrows = dem.height
    ncols = dem.width

    # Create a landlab uniform rectilinear grid and fill the nodes with elevation data
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=cell_size,
                        xy_of_lower_left=(dem.x.min(),dem.y.min().min()))
    zr = mg.add_zeros("topographic__elevation", at="node")
    zr += dem.data[::-1,:].ravel()

    # Estimate the flow direction and identify depressions and sinks
    mg.set_nodata_nodes_to_closed(zr, -9999)
    fa = FlowAccumulator(mg, flow_director="D4")
    fa.run_one_step()
    df = DepressionFinderAndRouter(mg)
    df.map_depressions()

    # Extract the data from the depression array (da) 
    da = mg.as_dataset("at_node:drainage_area")
    da = da["at_node:drainage_area"].data
    da = np.reshape(da, (nrows,ncols,), order="A")
    da = np.flipud(da)

    # Create the mask and buffer it by n-cells in all directions (mask_buf)
    da_mask= np.where(da>=min_chn_sz,1,0 )
    chn_msk = binary_dilation(da_mask, structure=np.ones((mask_buf, mask_buf))).astype(int)
    chn_edg_msk = binary_dilation(da_mask, structure=np.ones((mask_buf+1, mask_buf+1))).astype(int) # Get the edges of the channel
    # Create an edge mask and extract the hillslope elevation on the edges
    edges_mask = chn_edg_msk - chn_msk
    edges_data = np.where(edges_mask, dem.data, 0)

    # Smooth the elevation map, and use the mask to restrict smoothed value to within drainage basin
    chn_intrp = interpolate_mask(chn_msk, edges_data)
    smoothed_map = np.where(chn_msk, chn_intrp, dem.data)
    smoothed_map = gaussian_filter(smoothed_map, sigma=sigma)    # Smooth the output a bit
    smoothed_map = smoothed_map - elv_diff
    smoothed_map = np.where(chn_msk, smoothed_map, dem.data)
    smoothed_map = gaussian_filter(smoothed_map, sigma=1)
    smoothed_map = np.where(smoothed_map<0, 0, smoothed_map)

    # smoothed_map = gaussian_filter(dem.data, sigma=sigma)
    # smoothed_map = smoothed_map - elv_diff
    # smoothed_map = np.where(da_mask, smoothed_map, dem.data)
    # smoothed_map = gaussian_filter(smoothed_map, sigma=1)
    # smoothed_map = np.where(smoothed_map<0, 0, smoothed_map)

    # Update the dem data with the smoothed map and save the file
    dem.data = smoothed_map
    dem.to_netcdf(out_file)

    return dem

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert DEM GeoTIFF to xarray and optionally save as NetCDF")
    parser.add_argument("--input_file", help="Path to the input DEM NetCDF file")
    parser.add_argument("--out_file", "-o", help="Path to save the output NetCDF file (optional)")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the function with provided arguments
    smooth_dem_drainage(args.input_file, args.out_file)