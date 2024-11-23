import numpy as np

def fp(lambda_angle, delta, i_xi, phi, phi_s):
    """
    Calculate the P-wave radiation pattern based on the formula from Aki and Richards.
    
    Parameters:
        lambda_angle (float): Slip angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Source azimuth angle (ϕₛ) in radians.
    
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
        lambda_angle (float): Slip angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Source azimuth angle (ϕₛ) in radians.
    
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
        lambda_angle (float): Slip angle (λ) in radians.
        delta (float): Dip angle (δ) in radians.
        i_xi (float): Takeoff angle (iₓᵢ) in radians.
        phi (float): Azimuth angle (ϕ) in radians.
        phi_s (float): Source azimuth angle (ϕₛ) in radians.
    
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