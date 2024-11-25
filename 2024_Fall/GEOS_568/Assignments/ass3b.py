import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream
from obspy import UTCDateTime as OTC
from obspy.clients.fdsn import Client
from obspy.geodetics.base import degrees2kilometers, locations2degrees
from obspy.taup import TauPyModel
from pyrocko import orthodrome
import pygmt

client = Client("IRIS")
fc= 0.1

def stfunc(freq, total_duration, fc=1):
    """
    Generate a source time function (first derivative of a Gaussian).

    Parameters:
    frequency (float): Frequency of the timeseries in Hz.
    total_duration (int): Length of the timeseries in seconds.
    fc (float, optional): Corner frequency of the source in Hz. Defaults to 1

    Returns:
    np.ndarray: The source time function as a 1D NumPy array.
    """
    nsamps = total_duration * freq
    dt = 1/freq

    # Compute parameters
    period = 1.0 / fc        # Period of the source
    t0 = period / dt         # Time shift for peak alignment
    sigma = 4.0 / period     # Width of the Gaussian

    # Generate time array
    # time = np.arange(nsamps) * dt
    shifted_time = (np.arange(nsamps) - t0) * dt

    # Calculate the first derivative of the Gaussian
    stf = -2 * sigma * shifted_time * np.exp(-(sigma * shifted_time)**2)
    
    return stf


def u_dot_func(t, arrival_time, Mo=1):
    """
    Calculate a heaviside source time function
    Mo = scalar moment. Already used in displacement amplitude, set to 1.
    """
    dt = t[2]-t[1]
    M0 = lambda t: 0.5*Mo*0.5*(np.sign(t) + 1)
    G = np.diff(M0(t - arrival_time))/dt
    G = np.append(G, G[-1]) # Maintain the same length

    # x = (int(arrival_time//dt))   # Find the index of the arrival time
    # G= np.zeros_like(t)      # initialization G with zeros
    # if x <= len(t)-1:
    #     G[x] = 1
    return G


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
    term1 = np.sin(lambda_angle) * np.cos(2 * delta) * np.cos(2 * i_xi) * np.sin((phi - phi_s))
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
    term2 = np.cos(lambda_angle) * np.sin(delta) * np.sin(i_xi) * np.cos(2 * (phi - phi_s))
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

    # Compute the far-field displacement
    u_p = (F_P * mu * A) / (4 * np.pi * rho * alpha**3 * r) * u_dot_func(t, r/alpha) * l

    frq = 1 / (t[2]-t[1])
    stf = stfunc(frq, len(t),fc)
    # Convolution of Green's function with the source time function
    u_p= np.convolve(u_p, stf)[:len(u_p)]
    
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
    
    # Compute the vertical S-wave displacement (vectorized if t is an array)
    u_sv = (F_SV * mu * A) / (4 * np.pi * rho * beta**3 * r) * u_dot_func(t, r/beta) * p_hat

    frq = 1 / (t[2]-t[1])
    stf = stfunc(frq, len(t),fc)
    # Convolution of Green's function with the source time function
    u_sv= np.convolve(u_sv, stf)[:len(u_sv)]
    
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

    # Compute the horizontal S-wave displacement (vectorized if t is an array)
    u_sh = (F_SH * mu * A) / (4 * np.pi * rho * beta**3 * r) * u_dot_func(t, r/beta) * phi_hat

    frq = 1 / (t[2]-t[1])
    stf = stfunc(frq, len(t),fc)
    # Convolution of Green's function with the source time function
    u_sh= np.convolve(u_sh, stf)[:len(u_sh)]
    
    # Return scalar if t is scalar, otherwise array
    return u_sh if np.ndim(t) > 0 else u_sh[0]


def get_fiji_event_data(max_time=420,comp="BHZ"):
    """
    Download waveform data for 4 stations around the 2020 M5.9 fiji earthquake
    https://earthquake.usgs.gov/earthquakes/eventpage/us6000c617/executive

    The event waveforms have been demeaned, and the instrument response has 
    been removed to obtain displacement

    The function calculates the epicentral and hypocentral distances, azimuth, 
    p- and s- arrival times and take-off angles using ak135 1D vel. mod.

    Args:
        max_time(int, optional): time to crop waveform (s) after event origin
        comp (str, optional): Event component type. Defaults to "BHZ".

    Returns:
        dict: A dictionary with 3 main keys ["event","stations","waveforms"]
    """
    ######## Get the event
    starttime = OTC("2020-10-06 10:11:44")
    endtime = OTC("2020-10-06 10:12:00")
    cat = client.get_events(starttime=starttime,endtime=endtime,minmagnitude=5,
                            longitude=-178.472,latitude=-17.996,maxradius=1,)
    mag = cat[0].magnitudes[0].mag
    cat = cat[0].origins[0]
    elon,elat,edep,etime = cat.longitude,cat.latitude,cat.depth/1000,cat.time
    evt = {"lon":elon,"lat":elat,"dep":edep,"mag":mag,"etime":etime}
    ######## Get the stations
    invt = client.get_stations(longitude=elon, latitude=elat,startbefore=starttime,
                                minradius=7,maxradius=15,level="response",channel=comp)
    ######## Get the waveforms
    strm = Stream()
    stvt = {"net":[],"sta":[],"lon":[],"lat":[],"elv":[]}
    for ntwk in invt:
        for sta in ntwk:
            try:
                wv_fm = client.get_waveforms(ntwk.code, sta.code, "*", comp, etime, 
                                            etime+max_time, attach_response=True)
                stid = f"{ntwk.code}.{sta.code}"
                if len(wv_fm) > 0 and stid not in ["IU.FUNA","IU.RAO"]:
                    stvt["net"].append(ntwk.code),stvt["sta"].append(sta.code)
                    stvt["lon"].append(sta.longitude),stvt["lat"].append(sta.latitude)
                    stvt["elv"].append(sta.elevation/1000) # m to km
                    strm += wv_fm[0]
            except:
                pass
    ######## Calculate the hypocentral distances for the waveforms
    vmod = TauPyModel(model="iasp91")
    stvt = pd.DataFrame(stvt)
    stvt["epd"] = stvt.apply(lambda rw: degrees2kilometers(locations2degrees(rw.lat, rw.lon, elat, elon)), axis=1)
    stvt["hyd"] = stvt.apply(lambda rw: np.sqrt(((edep+rw.elv)**2 + rw.epd**2)), axis=1)
    stvt["azi"] = stvt.apply(lambda rw: orthodrome.azimuth(elat, elon, rw.lat, rw.lon,), axis=1) # stn to rcv

    df_arr = []
    for j,rw in stvt.iterrows():
        rlat, rlon = rw.lat,rw.lon
        # _,az,_ = gps2dist_azimuth(elat, elon, rw.lat, rw.lon)
        arrivals = vmod.get_ray_paths_geo(edep,elat,elon,rlat,rlon,phase_list=["p","P","s","S"])
        p_ts,p_tko,s_ts,s_tko = [],[],[],[]
        for arv in arrivals:
            if arv.name.lower() == "p":
                p_ts.append(arv.time), p_tko.append(arv.takeoff_angle)
            elif arv.name.lower() == "s":
                s_ts.append(arv.time), s_tko.append(arv.takeoff_angle)
        line = [rw.net, rw.sta, p_ts[0], p_tko[0],s_ts[0],s_tko[0]]
        df_arr.append(line)
    df_arr = pd.DataFrame(df_arr,columns=["net","sta","p_ts","p_tko","s_ts","s_tko"])

    stvt = stvt.merge(df_arr,on=["net","sta"])

    flt_cols = ["elv","epd","hyd","azi","p_ts","p_tko","s_ts","s_tko"]
    stvt[flt_cols] = stvt[flt_cols].round(2)
    ######## Remove instrument response, convert to displacement
    strm= strm.detrend("demean")
    strm = strm.remove_response(output="DISP")

    output = {"event":evt,
              "stations":stvt,
              "waveforms":strm}

    # print(stvt)
    # strm= strm.filter("lowpass",freq=1)
    # strm.plot(equal_scale=False);
    # print(strm)

    return output



if __name__=="__main__":
    save_dir = "output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Fiji earthquake data
    strike = 183       # Strike angle (degrees)
    dip = 83           # Dip angle (degrees)
    rake = 75          # Rake angle (degrees)
    mu = 1.046e15      # Shear modulus (Pa) - made it equiv to moment magnitude of event
    A = 1              # Fault area (km²) because M = μAD
    rho = 2.7          # Density (kg/m³)
    alpha = 9.5        # P-wave speed (km/s)
    beta = 4.9         # S-wave speed (km/s)

    data = get_fiji_event_data()
    evt = data["event"]
    stns = data["stations"]
    wvfms = data["waveforms"]

    # Plot the stations relative to the hypocenter
    fig = pygmt.Figure()
    fig.coast(frame="afg",region="g",projection="G-178.5/-18/12c",land="khaki",water="white",)
    fig.meca(spec={"strike": 183, "dip": 83, "rake": 75, "magnitude": 5.9},scale="0.5c",
             longitude=evt["lon"],latitude=evt["lat"],depth=evt["dep"],
             compressionfill="red",extensionfill="white",pen="0.5p,gray30,solid",)
    stids = (stns["net"]+"."+stns["sta"]).to_list()
    fig.plot(x=stns["lon"], y=stns["lat"], style="i0.29c", fill="seagreen", pen="1p,black")
    fig.text(x=stns["lon"]+6, y=stns["lat"]-1, text=stids, font="6p,Times-Bold")
    fig.savefig(f"{save_dir}/station_map.png")

    for idx,row in stns.iterrows():
        trc = wvfms[idx]
        # Takeoff angle (degrees)
        p_xi = row.p_tko   
        s_xi = row.s_tko  
        phi = row.azi           # Azimuth angle (degrees)  
        hyd = row.epd           # hypocentral Distance (km)  
        
        # Calculate far-field displacement
        t = np.linspace(0,420,trc.stats.npts,endpoint=False)  # Observation time (s)
        u_p = disp_p(0, t, rake, dip, p_xi, phi, strike, mu, A, rho, alpha, hyd, u_dot_func)
        u_sv = disp_sv(0, t, rake, dip, s_xi, phi, strike, mu, A, rho, beta, hyd, u_dot_func)
        u_sh = disp_sh(0, t, rake, dip, s_xi, phi, strike, mu, A, rho, beta, hyd, u_dot_func)
        syn_disp = u_p + u_sv + u_sh # Add the shear wave hor. and ver. components

        fig,ax= plt.subplots(2,1,figsize=(9,4), sharex=True)
        ax[0].plot(t,trc.data,c="k",lw=1),ax[0].margins(x=0.01)
        ax[0].axvline(row.p_ts,c="r",lw=1,ymin=0.25,ymax=0.75)
        ax[0].axvline(row.s_ts,c="b",lw=1,ymin=0.25,ymax=0.75)
        ax[1].plot(t,syn_disp,c="k",ls="--",label=r"$u^T$")   # Plot a combined synthetic waveform
        ax[1].plot(t,u_p,c="r",ls="--",label="P",alpha=0.75)
        ax[1].plot(t,u_sv,c="orange",ls="--",label="SV",alpha=0.75)
        ax[1].plot(t,u_sh,c="cyan",ls="--",label="SH",alpha=0.75)
        ax[1].legend(loc="upper right", handlelength=0.5)
        ax[1].set_xlabel("Time (s)")
        stid = f"{row.net}.{row.sta}"
        ax[0].set_title(stid)
        plt.savefig(f"{save_dir}/{stid}.png",dpi=200,bbox_inches="tight")
        plt.close()

    