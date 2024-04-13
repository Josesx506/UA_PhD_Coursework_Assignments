import arviz as az
import numpy as np
import pymc3 as pm
from tqdm import tqdm
seed = 320
np.random.seed(seed)


def synthetic_seis(length,start_idx,freq=10):
    start_idx = np.floor(start_idx).astype(int)*freq
    idxs = np.array([3,2,4,3,7,3])*(freq)
    csum = np.cumsum(idxs)
    array = np.zeros(length)

    if start_idx+csum[-1]<length:
        array[start_idx:start_idx+csum[0]] = np.random.normal(0,4,idxs[0])
        array[start_idx+csum[0]:start_idx+csum[1]] = np.random.normal(0,2,idxs[1])
        array[start_idx+csum[1]:start_idx+csum[2]] = np.random.normal(0,8,idxs[2])
        array[start_idx+csum[2]:start_idx+csum[3]] = np.random.normal(0,3,idxs[3])
        array[start_idx+csum[3]:start_idx+csum[4]] = np.random.normal(0,12,idxs[4])
        array[start_idx+csum[4]:start_idx+csum[5]] = np.random.normal(0,6,idxs[5])
        array = (array.max() - array)/(array.max() - array.min())
        return array
    else:
        raise ValueError("Start time should be shorter than length waveforms")


def predicted_travel_time(source_loc, station_loc, velocity):
    distances = np.sqrt(np.sum(np.square(station_loc - source_loc), axis=1))
    return distances / velocity


def likelihood(Tpred,Tobs,sigma,ix,iy):
    """
    Equal Differential time likelihood function
    """
    dtp    = (Tpred[ix]-Tpred[iy])
    dto    = (Tobs[ix]-Tobs[iy])
    dsigma = (sigma[ix]**2 + sigma[iy]**2)
    logp   = abs((dtp - dto)/(np.sqrt(dsigma)))*np.sqrt(2) + np.log(1/(dsigma*np.sqrt(2)))
    logp   = np.nansum(logp,axis=0)
    return -logp


def simulate_loss(params, Xmin, Xmax, Zmin, Zmax, Xc, Vh, ix, iy, nsim=10000, n_samples=1000):
    sim_loss = []
    sim_Xeqloc = []
    sim_Zeqloc = []

    for itr in tqdm(range(nsim),position=0):
        Xesc = np.random.uniform(Xmin, Xmax)
        Zesc = np.random.uniform(Zmin, Zmax)

        Tobs = params["Tobs"]                          # Ground truth travel times
        Tobs_sigma = params["Tobs_sigma"]              # Travel time noise errors
        Tpred_Params = params["Tpred_Params"]

        Xst = Xc[:,0] # Station x-location
        Zst = Xc[:,1] # Station z-location

        Tpred = predicted_travel_time(np.array([Xesc,Zesc]), np.stack([Xst,Zst], axis=1), Vh)

        SigmaP = np.clip(Tpred * Tpred_Params[0], Tpred_Params[1], Tpred_Params[2])

        SigmaO = Tobs_sigma
        sigma  = np.sqrt(SigmaO**2 + SigmaP**2)
        like = likelihood(Tpred, Tobs, sigma, ix, iy)

        predXesc = [Xesc]
        predZesc = [Zesc]
        loss_estimates = [like]

        # Perform a single realization with n_samples
        for i in range(1, n_samples):
            Xesc = np.random.uniform(Xmin, Xmax)
            Zesc = np.random.uniform(Zmin, Zmax)

            Tobs = params["Tobs"]                          # Ground truth travel times
            Tobs_sigma = params["Tobs_sigma"]              # Travel time noise errors
            Tpred_Params = params["Tpred_Params"]

            Xst = Xc[:,0] # Station x-location
            Zst = Xc[:,1] # Station z-location

            Tpred = predicted_travel_time(np.array([Xesc,Zesc]), np.stack([Xst,Zst], axis=1), Vh)

            SigmaP = np.clip(Tpred * Tpred_Params[0], Tpred_Params[1], Tpred_Params[2])

            SigmaO = Tobs_sigma
            sigma  = np.sqrt(SigmaO**2 + SigmaP**2)
            new_like = likelihood(Tpred, Tobs, sigma, ix, iy)

            div = new_like / like  # Compare the current and prior errors

            u = np.random.uniform(size=1)

            if div < u:
                like = new_like
                loss_estimates.append(like)
                predXesc.append(Xesc)
                predZesc.append(Zesc)
            else:
                loss_estimates.append(loss_estimates[i-1])
                predXesc.append(predXesc[i-1])
                predZesc.append(predZesc[i-1])

        # Append only the last items
        sim_loss.append(loss_estimates[-1])
        sim_Xeqloc.append(predXesc[-1])
        sim_Zeqloc.append(predZesc[-1])

    return {"sim_loss": sim_loss, 
            "sim_Xeqloc": sim_Xeqloc, 
            "sim_Zeqloc": sim_Zeqloc,
            "last_rlz_loss":loss_estimates,
            "last_rlz_Xeqloc":predXesc,
            "last_rlz_Zeqloc":predZesc}

# Example usage:
# sim_data = simulate_loss(params, Xmin, Xmax, Zmin, Zmax, Xc, Vh, obs_sigma, likelihood, predicted_travel_time)


def MCMC(likelihood,Params,vel,ix,iy,draws=10000,tuning=1400,num_drawn_samples=1000,RANDOM_SEED=seed):
    with pm.Model() as LocModel:
        Xe     = pm.Uniform("Earthquake X Location",lower=float(Params["X_Bounds"][0]),upper=float(Params["X_Bounds"][1]))
        Ze     = pm.Uniform("Earthquake Z Location",lower=float(Params["X_Bounds"][0]),upper=float(Params["Z_Bounds"][1]))

        # Defining the model predicted time uncertainties
        Tpred_Params = pm.Data("Tpred_Params",Params["Tpred_Params"])

        # Defining Observational Phase Pick Times & gaussian std uncertainties      
        Tobs       = pm.Data("Tobs",Params["Tobs"])
        Tobs_sigma = pm.Data("Tobs_sigma",Params["Tobs_sigma"])
        
        # Defining Station Locations 
        Xst    = pm.Data("Xst",Params["X_station"][:,0])
        Zst    = pm.Data("Zst",Params["X_station"][:,1])
        
        # --- Determining the predicted travel-time ---
        Tpred  = (np.sqrt((Xst-Xe)**2 + (Zst-Ze)**2))/vel

        # --- Determining the Posterior Uncertainty ---
        SigmaP = pm.math.clip(Tpred*Tpred_Params[0],Tpred_Params[1],Tpred_Params[2])
        
        SigmaO = Tobs_sigma
        sigma  = np.sqrt(SigmaO**2 + SigmaP**2)
          
        # -- Defining the likelihood function --
        like = pm.Potential("like",likelihood(Tpred,Tobs,sigma,ix,iy))

        # -- Hamiltonian Monte Carlo Sampling --
        print("-----------------------------------------------------")
        print("----------- MCMC Parameter Estimation ---------------")
        step = pm.HamiltonianMC()
        trace = pm.sample(draws=draws,tune=tuning,step=step) # ,return_inferencedata=True

        # -- Drawing Samples from the recovered distribution --
        print("-----------------------------------------------------")
        print("--------  Drawing Samples from Distribution -------- ")
        drawn_samples = pm.sample_posterior_predictive(trace, var_names=["Earthquake X Location", "Earthquake Z Location"], random_seed=RANDOM_SEED,size=num_drawn_samples)

        # -- Plotting the distribution in the earthquake location and summary stats --
        print("-----------------------------------------------------")
        print("-------------  Plotting and Summary  --------------- ")
        pm.plots.traceplot(trace)
        summary = az.summary(trace, kind="stats")

    return trace,summary,drawn_samples

