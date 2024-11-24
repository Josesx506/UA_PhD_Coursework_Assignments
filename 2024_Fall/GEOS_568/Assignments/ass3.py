import numpy as np
import matplotlib.pyplot as plt

# Define a source time function (e.g., Gaussian pulse derivative)
def u_dot_func(t):
    slip = np.exp(-t**2)
    slip = np.where(t<0,0,slip)
    return slip

def fp(lambda_angle, delta, i_xi, phi, phi_s):
    """
    Calculate the P-wave radiation pattern based on the formula from Aki and Richards.
    
    Parameters:
        lambda_angle (float): Rake angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Strike angle (ϕₛ) in radians.
    
    Returns:
        float: The P-wave radiation pattern (F^P).
    """
    # Compute individual terms
    term1 = np.cos(lambda_angle) * np.sin(delta) * (np.sin(i_xi) ** 2) * np.sin(2 * (phi - phi_s))
    term2 = -np.cos(lambda_angle) * np.cos(delta) * np.sin(2 * i_xi) * np.cos(phi - phi_s)
    term3 = np.sin(lambda_angle) * np.sin(2 * delta) * (np.cos(i_xi) ** 2 - (np.sin(i_xi) ** 2) * (np.sin(phi - phi_s) ** 2))
    term4 = np.sin(lambda_angle) * np.cos(2 * delta) * np.sin(2 * i_xi) * np.sin(phi - phi_s)
    
    # Combine terms
    fp_value = term1 + term2 + term3 + term4
    
    return fp_value


def fsv(lambda_angle, delta, i_xi, phi, phi_s):
    """
    Calculate the vertical S-wave (SV) radiation pattern based on the given formula.
    
    Parameters:
        lambda_angle (float): Rake angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Strike angle (ϕₛ) in radians.
    
    Returns:
        float: The vertical S-wave (SV) radiation pattern (F^SV).
    """
    # Compute individual terms
    term1 = np.sin(lambda_angle) * np.cos(2 * delta) * np.cos(2 * i_xi) * np.sin(2 * (phi - phi_s))
    term2 = -np.cos(lambda_angle) * np.cos(delta) * np.cos(2 * i_xi) * np.cos(phi - phi_s)
    term3 = 0.5 * np.cos(lambda_angle) * np.sin(delta) * np.sin(2 * i_xi) * np.sin(2 * (phi - phi_s))
    term4 = -0.5 * np.sin(lambda_angle) * np.sin(2 * delta) * np.sin(2 * i_xi) * (1 + (np.sin(phi - phi_s) ** 2))
    
    # Combine terms
    fsv_value = term1 + term2 + term3 + term4
    
    return fsv_value

def fsh(lambda_angle, delta, i_xi, phi, phi_s):
    """
    Calculate the horizontal S-wave (SH) radiation pattern based on the given formula.
    
    Parameters:
        lambda_angle (float): Rake angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Strike angle (ϕₛ) in radians.
    
    Returns:
        float: The horizontal S-wave (SH) radiation pattern (F^SH).
    """
    # Compute individual terms
    term1 = np.cos(lambda_angle) * np.cos(delta) * np.cos(i_xi) * np.sin(phi - phi_s)
    term2 = -np.cos(lambda_angle) * np.sin(delta) * np.sin(i_xi) * np.cos(2 * (phi - phi_s))
    term3 = np.sin(lambda_angle) * np.cos(2 * delta) * np.cos(i_xi) * np.cos(phi - phi_s)
    term4 = -0.5 * np.sin(lambda_angle) * np.sin(2 * delta) * np.sin(i_xi) * np.sin(2 * (phi - phi_s))
    
    # Combine terms
    fsh_value = term1 + term2 + term3 + term4
    
    return fsh_value


def disp_p(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, alpha, r, u_dot_func, l=1):
    """
    Calculate the far-field displacement for the P-wave.
    
    Parameters:
        x (float): Position coordinate.
        t (float): Time.
        lambda_angle (float): Rake angle (λ) in degrees.
        delta (float): Dip angle (δ) in degrees.
        i_xi (float): Takeoff angle (iₓᵢ) in degrees.
        phi (float): Azimuth angle (ϕ) in degrees.
        phi_s (float): Strike angle (ϕₛ) in degrees.
        mu (float): Shear modulus (Pa).
        A (float): Fault area (m²).
        rho (float): Density (kg/m³).
        alpha (float): P-wave speed (m/s).
        r (float): Epicentral distance from source to observation point (m).
        u_dot_func (function): Function for \( \dot{\overline{u}} \), the time-dependent source term.
    
    Returns:
        float: Far-field displacement \( u^p(x, t) \).
    """
    # Convert angles from degrees to radians
    lambda_angle_rad = np.deg2rad(lambda_angle)
    delta_rad = np.deg2rad(delta)
    i_xi_rad = np.deg2rad(i_xi)
    phi_rad = np.deg2rad(phi)
    phi_s_rad = np.deg2rad(phi_s)
    
    # Calculate F^P using the fp() function
    F_P = fp(lambda_angle_rad, delta_rad, i_xi_rad, phi_rad, phi_s_rad)
    
    # Time shift term
    t_shift = t - r / alpha

    # Compute the far-field displacement
    u_p = (F_P * mu * A) / (4 * np.pi * rho * alpha**3 * r) * u_dot_func(t_shift) * l
    
    return u_p

def disp_sv(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, beta, r, u_dot_func, p_hat=1):
    """
    Calculate the vertical S-wave displacement (SV-wave) at a point.

    Parameters:
        x (float): Position coordinate.
        t (float): Time.
        lambda_angle (float): Rake angle (λ) in degrees.
        delta (float): Dip angle (δ) in degrees.
        i_xi (float): Takeoff angle (iₓᵢ) in degrees.
        phi (float): Azimuth angle (ϕ) in degrees.
        phi_s (float): Strike angle (ϕₛ) in degrees.
        mu (float): Shear modulus (Pa).
        A (float): Fault area (m²).
        rho (float): Density (kg/m³).
        beta (float): S-wave speed (m/s).
        r (float): Epicentral distance from source to observation point (m).
        u_dot_func (function): Function for \( \dot{\overline{u}} \), the time-dependent source term.
        p_hat (float): Directional factor for SV-wave (default is 1).

    Returns:
        float or array-like: Far-field displacement \( u^{SV}(x, t) \) for the given time(s).
    """
    # Convert angles from degrees to radians
    lambda_angle_rad = np.deg2rad(lambda_angle)
    delta_rad = np.deg2rad(delta)
    i_xi_rad = np.deg2rad(i_xi)
    phi_rad = np.deg2rad(phi)
    phi_s_rad = np.deg2rad(phi_s)
    
    # Calculate F^{SV} using the fsv() function
    F_SV = fsv(lambda_angle_rad, delta_rad, i_xi_rad, phi_rad, phi_s_rad)
    
    # Time shift term
    t_shift = t - r / beta
    
    # Compute the vertical S-wave displacement (vectorized if t is an array)
    u_sv = (F_SV * mu * A * p_hat) / (4 * np.pi * rho * beta**3 * r) * u_dot_func(t_shift) * p_hat
    
    # Return scalar if t is scalar, otherwise array
    return u_sv if np.ndim(t) > 0 else u_sv[0]

def disp_sh(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, beta, r, u_dot_func, phi_hat=1):
    """
    Calculate the horizontal S-wave displacement (SH-wave) at a point.

    Parameters:
        x (float): Position coordinate.
        t (float or array-like): Time or array of time values.
        lambda_angle (float): Slip angle (λ) in degrees.
        delta (float): Dip angle (δ) in degrees.
        i_xi (float): Takeoff angle (iₓᵢ) in degrees.
        phi (float): Azimuth angle (ϕ) in degrees.
        phi_s (float): Strike angle (ϕₛ) in degrees.
        mu (float): Shear modulus (Pa).
        A (float): Fault area (m²).
        rho (float): Density (kg/m³).
        beta (float): S-wave speed (m/s).
        r (float): Epicentral distance from source to observation point (m).
        u_dot_func (function): Function for \( \dot{\overline{u}} \), the time-dependent source term.
        phi_hat (float): Directional factor for SH-wave (default is 1).

    Returns:
        float or array-like: Far-field displacement \( u^{SH}(x, t) \) for the given time(s).
    """
    # Convert angles from degrees to radians
    lambda_angle_rad = np.deg2rad(lambda_angle)
    delta_rad = np.deg2rad(delta)
    i_xi_rad = np.deg2rad(i_xi)
    phi_rad = np.deg2rad(phi)
    phi_s_rad = np.deg2rad(phi_s)
    
    # Calculate F^{SH} using the fsh() function
    F_SH = fsh(lambda_angle_rad, delta_rad, i_xi_rad, phi_rad, phi_s_rad)
    
    # Time shift term
    t_shift = t - r / beta
    
    # Compute the horizontal S-wave displacement (vectorized if t is an array)
    u_sh = (F_SH * mu * A * phi_hat) / (4 * np.pi * rho * beta**3 * r) * u_dot_func(t_shift) * phi_hat
    
    # Return scalar if t is scalar, otherwise array
    return u_sh if np.ndim(t) > 0 else u_sh[0]



if __name__=="__main__":
    # Example parameters
    lambda_angle = 30  # Rake angle (degrees)
    delta = 45         # Dip angle (degrees)
    i_xi = 60          # Takeoff angle (degrees)
    phi = 90           # Azimuth angle (degrees)
    phi_s = 45         # Strike angle (degrees)
    mu = 3e10          # Shear modulus (Pa)
    A = 1e6            # Fault area (m²)
    rho = 2700         # Density (kg/m³)
    alpha = 6000       # P-wave speed (m/s)
    beta = 3500        # S-wave speed (m/s)
    r = 10e3           # hypocentral Distance (m)
    x = 0#1.6e6        # Latitude distance from fault (m) - This isn't used in the function

    # Calculate far-field displacement
    t = np.linspace(-5,10,200)  # Observation time (s)
    u_p = disp_p(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, alpha, r, u_dot_func)
    u_sv = disp_sv(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, beta, r, u_dot_func)
    u_sh = disp_sh(x, t, lambda_angle, delta, i_xi, phi, phi_s, mu, A, rho, beta, r, u_dot_func)
    # print("Far-field P-wave displacement:", u_p_value)

    # plt.plot(t,u_p)
    # plt.plot(t,u_sv)
    plt.plot(t,u_sh)
    plt.show()