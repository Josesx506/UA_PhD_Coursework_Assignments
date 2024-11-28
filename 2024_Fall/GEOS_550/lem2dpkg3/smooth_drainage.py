import argparse

import numpy as np
import xarray as xr
from landlab import RasterModelGrid
from landlab.components import DepressionFinderAndRouter, FlowAccumulator
from scipy.ndimage import binary_dilation, gaussian_filter


def smooth_dem_drainage(input_file:str,
                        out_file:str,
                        cell_size: int = 100,
                        min_chn_sz:float = 1e6,
                        sigma:int = 10,
                        mask_buf:int = 7,
                        elv_diff:int = 100
                        ):
    """
    Given a DEM map, estimate the drainage area. Smooth the topography only within 
    the drainage basins (channels) and drop the elevation by a limited height.
    This is useful for providing input conditions for knickpoints where the smoothed 
    channels serve as pre-erosion topography. The elevation is also used to ensure 
    we don't have to deal with uplifting topography from 0 during the landscape 
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
    nrows = dem.height
    ncols = dem.width

    # Create a landlab uniform rectilinear grid and fill the nodes with elevation data
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=cell_size,
                        xy_of_lower_left=(dem.x.min(),dem.y.min().min()))
    zr = mg.add_zeros("topographic__elevation", at="node")
    zr += dem.data[::-1,:].ravel()

    # Estimate the flow direction and identify depressions and sinks
    mg.set_nodata_nodes_to_closed(zr, -9999)
    fa = FlowAccumulator(mg, flow_director='D4')
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
    da_mask = binary_dilation(da_mask, structure=np.ones((mask_buf, mask_buf))).astype(int)
    da_mask = gaussian_filter(da_mask, sigma=0.25)

    # Smooth the elevation map, and use the mask to restrict smoothed value to within drainage basin
    smoothed_map = gaussian_filter(dem.data, sigma=sigma)
    smoothed_map = np.where(da_mask, smoothed_map, dem.data)
    smoothed_map = smoothed_map - elv_diff
    smoothed_map = np.where(smoothed_map<0, 0, smoothed_map)

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