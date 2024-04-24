import laspy
import mplstereonet as st
import numpy as np
from numpy import sqrt,power,pi,radians,degrees,sin,cos
from pyproj import CRS, Transformer

def dip_direction_to_strike(dip_direction: int):
    '''
    Convert dip direction to strike
        Makes the strike data compatible with the lower hemisphere right-hand-rule
        that mplstereonets expects
    '''
    strike = (dip_direction - 90) % (360)
    return strike

def mean_dilation_angle(k, lim=50):
    '''
    Calculate dilation angle from fisher statistics kappa
    '''
    if lim == 50:
        da = 67.5 / np.sqrt(k)
        return da
    elif lim == 63:
        da = 81 / np.sqrt(k)
        return da
    elif lim == 95:
        da = 140 / np.sqrt(k)
        return da
    else:
        raise ValueError("Input limit value should be one of 50, 63, or 95")

def calc_planar_daylight(strike,dip):
    """
    Draws the planar daylight envelope (cone) with respect to a 
    slope face with a given strike and dip.

    Source: https://github.com/ricsatjr/rockinematics
    
    Parameters
    ----------
    strike : number or sequence of numbers
        The strike of the plane(s) in degrees, with dip direction indicated by
        the azimuth (e.g. 315 vs. 135) specified following the "right hand
        rule".
    dip : number or sequence of numbers
        The dip of the plane(s) in degrees.
        
    Returns
    -------
    pde_plunge, pde_bearing, pde_angle: arrays
        Arrays of plunges, bearings, and angles of the planar daylight envelopes (cones).
    """

    strike, dip = np.atleast_1d(strike, dip)
    # calculating plunge and bearing of pole to plane
    p_plunge, p_bearing=st.pole2plunge_bearing(strike, dip)
    # calculating plunge, bearing, and angle of planar daylight envelope (cone)
    pde_plunge=45+p_plunge/2.
    pde_bearing=p_bearing
    pde_angle=45-p_plunge/2.-10**-9

    return np.around(pde_plunge,2), np.around(pde_bearing,2), np.around(pde_angle,2)


def convert_las_crs_to_projected_coords(filepath,input_crs,output_crs,inp_fmt=".laz",out_fmt=".las"):
    """
    Convert a lat and longitude coordinate reference system point cloud
    into a projected coordinate reference system. 
    The point cloud format should be .las or .laz
    The function saves the new file and returns None

    Args:
        filepath (str): path to the point cloud las file
        input_crs (int): EPSG code for input coordinate reference system
        output_crs (int): EPSG code for output coordinate reference system

    Returns:
        None

    Example
    convert_las_crs_to_projected_coords("CableMnt_RA_2019_Pointcloud.laz",4326,26912)
    """
    las = laspy.read(filepath)

    # Define original and desired CRS
    orig_crs = CRS(f"EPSG:{input_crs}")
    proj_crs = CRS(f"EPSG:{output_crs}")

    crs_trans = Transformer.from_crs(orig_crs, proj_crs, always_xy=True)
    x_trans, y_trans = crs_trans.transform(las.x,las.y)

    # Update coordinates in LAS file. Get the median center of the point cloud
    mx = np.quantile(x_trans,0.5)
    my = np.quantile(y_trans,0.5)
    las.header.offset = np.array([mx, my, 1.0000e+03])      #np.array([5.2620e+05, 3.5747e+06, 1.0000e+03])
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.x = x_trans
    las.y = y_trans

    las.header.add_crs(proj_crs)
    out_path = filepath.split("/")
    out_path[-1] = f"Proj_EPSG_{output_crs}_" + out_path[-1]
    out_path[-1] = out_path[-1].replace(inp_fmt,out_fmt)
    out_path = "/".join(out_path)

    las.write(out_path)

    return None


def distance_along_profile_proj_crs(x1, y1, x2, y2, num_points):
    """
    Calculate the distance along a profile between two sets of x and y points.
    
    Parameters:
        x1 (array-like): x-coordinates of the first set of points.
        y1 (array-like): y-coordinates of the first set of points.
        x2 (array-like): x-coordinates of the second set of points.
        y2 (array-like): y-coordinates of the second set of points.
        num_points (int): Number of points to discretize the profile.
        
    Returns:
        distances (array): Array containing the distance along the profile at each discretized point.
    """
    # Interpolate points along the profile
    x_interp = np.linspace(x1, x2, num_points)
    y_interp = np.linspace(y1, y2, num_points)
    
    # Calculate the distance between adjacent points
    dx = np.diff(x_interp)
    dy = np.diff(y_interp)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Calculate cumulative distance
    cumulative_distances = np.cumsum(distances)
    
    return cumulative_distances


def sigma_stress_theta1(k1,r,theta):
    """
    Estimate stress for mode 1 stress intensity factors

    Different theta values are provided, and the angle with the maximum
    stress is the crack growth angle
    """
    mode1 = (k1 / sqrt(2*pi*r)) * power(cos(radians(theta/2)), 3)
    return mode1


def dip_to_slope(dip_angle):
    """
    Convert dip angle in degrees to slope.
    
    Parameters:
        dip_angle (float): The dip angle in degrees.
        
    Returns:
        slope (float): The slope of the line.
    """
    slope = np.tan(dip_angle * (np.pi / 180))
    return slope