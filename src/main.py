from UKF import UnscentedKalmanFilter
from create_synthetic_data import PulsarFrequencyObservations
from model import MelatosPTAModel


from configs.config import canonical as cfg
import numpy as np


#First, let's create some synthetic data.
dt   = cfg["timing_parameters"]["dt_days"] 
Tend = cfg["timing_parameters"]["T_years"]     
t    = np.arange(0.0,Tend*365*24*3600,dt*24*3600)

observations = PulsarFrequencyObservations(t) # initialise the class, all observations have same times
observations.create_observations(cfg["pulsar_parameters"],cfg["GW_parameters"],cfg["noise_parameters"])
#observations.plot_observations(psr_index=None) #Can plot this





#Now initialise the state-space model to be used with the UKF

dictionary_of_known_quantities = {"dec_psr":cfg["pulsar_parameters"]["dec_psr"],
                                  "ra_psr":cfg["pulsar_parameters"]["ra_psr"],
                                  "pulsar_distances":cfg["pulsar_parameters"]["pulsar_distances"]}
model = MelatosPTAModel(observations.Npulsars + 1,observations.Npulsars,dictionary_of_known_quantities)









#Now let's run the UKF on this data
n_states = observations.Npulsars + 1 #N psr frequencies + GW phase
KF = UnscentedKalmanFilter(n_states=n_states,
                           observations=observations,
                           model = model
                           )


parameters = {"omega":observations.omega_GW,
              "gamma":observations.spindown_gamma[0],
              "n":observations.spindown_n[0],
              "dec_gw":cfg["GW_parameters"]["dec_GW"],
              "ra_gw":cfg["GW_parameters"]["ra_GW"],
              "psi_gw":cfg["GW_parameters"]["psi_GW"],
              "Agw":observations.Agw,
              "iota_gw": cfg["GW_parameters"]["iota"]
             }
KF.ll_on_data(parameters)

