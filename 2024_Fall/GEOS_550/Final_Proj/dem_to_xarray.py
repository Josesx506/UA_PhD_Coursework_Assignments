import argparse
import numpy as np
import rasterio
import xarray as xr
import matplotlib.pyplot as plt


def dem_to_xarray(tif_path, output_path=None):
    """
    Convert a DEM GeoTIFF file to an xarray.DataArray and optionally save as a NetCDF.
    
    Args:
        tif_path (str): Path to the input DEM GeoTIFF file.
        output_path (str, optional): Path to save the output NetCDF file. If None, the file will not be saved.
    
    Returns:
        xarray.DataArray: The DEM data as an xarray object.
    """
    
    # Open the GeoTIFF DEM file
    with rasterio.open(tif_path) as src:
        dem_data = src.read(1)  # Read the first (or only) band into an array
        profile = src.profile    # Get metadata of the GeoTIFF
        transform = src.transform
        crs = src.crs  # Get the CRS (coordinate reference system)

    # Create x and y coordinate arrays from the affine transformation
    x_coords = transform[2] + transform[0] * (0.5 + np.arange(src.width))
    y_coords = transform[5] + transform[4] * (0.5 + np.arange(src.height))

    # Remove "nodata" from attributes if it is None
    if profile["nodata"] is None:
        profile.pop("nodata", None)  # Remove "nodata" if it is None to avoid serialization issues

    # Convert the CRS to WKT format (string) and add it to the attributes
    profile["crs"] = crs.to_wkt() if crs else None

    # Convert boolean attributes to int or remove them
    boolean_attrs = ['tiled', 'interleave', 'compress']  # Example attributes that may be booleans
    for attr in boolean_attrs:
        if attr in profile:
            if isinstance(profile[attr], bool):
                profile[attr] = int(profile[attr])  # Convert True/False to 1/0

    # Create the DataArray
    dem_xarray = xr.DataArray(
        dem_data,
        coords=[y_coords, x_coords],  # y-coordinates (latitude), x-coordinates (longitude)
        dims=["y", "x"],              # Dimension names corresponding to axes
        attrs=profile                 # Metadata from the GeoTIFF
    )

    

    # Print the xarray object for verification
    print(dem_xarray)

    fig,ax = plt.subplots(figsize=(7,5))
    dem_xarray.plot(ax=ax,aspect=None,cmap="terrain")
    plt.show()

    # Optionally save the DataArray to a NetCDF file
    if output_path:
        dem_xarray.to_netcdf(output_path)
        print(f"NetCDF file saved to {output_path}")

    return dem_xarray

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert DEM GeoTIFF to xarray and optionally save as NetCDF")
    parser.add_argument("tif_path", help="Path to the input DEM GeoTIFF file")
    parser.add_argument("--output", "-o", help="Path to save the output NetCDF file (optional)", default=None)
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call the function with provided arguments
    dem_to_xarray(args.tif_path, args.output)
