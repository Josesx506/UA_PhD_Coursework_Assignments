import mplstereonet as st
import numpy as np

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