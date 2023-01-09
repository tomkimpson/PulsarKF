from UKF import UnscentedKalmanFilter
from create_synthetic_data import PulsarFrequencyObservations

from configs.config import canonical as cfg
import numpy as np


#First, let's create some synthetic data.
dt   = cfg["timing_parameters"]["dt_days"] 
Tend = cfg["timing_parameters"]["T_years"]     

print(cfg)
t    = np.arange(0.0,Tend*365*24*3600,dt*24*3600)
observations = PulsarFrequencyObservations(t) # initialise the class, all observations have same times
observations.create_observations(cfg["pulsar_parameters"],cfg["GW_parameters"],cfg["noise_parameters"])


observations.plot_observations(psr_index=None)







#n_states = 3
#x = UnscentedKalmanFilter(n_states=n_states)


