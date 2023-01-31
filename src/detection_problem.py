from UKF import UnscentedKalmanFilter
from create_synthetic_data import PulsarFrequencyObservations
from model import MelatosPTAModel
from bayesian_inference import BilbySampler

from configs.config import canonical as cfg
import numpy as np
import bilby 
import os 




import sys




def run(pulsar_parameters,GW_parameters,noise_parameters,t):




    observations = PulsarFrequencyObservations(t)              # Initialise the class, all observations have same times

    observations.create_observations(pulsar_parameters,
                                     GW_parameters,
                                     noise_parameters)  # generate the observations. You can also plot this as e.g. observations.plot_observations(psr_index=2,KF_predictions = None) 
 
    


    # Now initialise the state-space model to be used with the UKF
    dictionary_of_known_quantities = {"dec_psr":         pulsar_parameters["dec_psr"],
                                      "ra_psr":          pulsar_parameters["ra_psr"],
                                      "pulsar_distances":pulsar_parameters["pulsar_distances"]}
    
    model = MelatosPTAModel(observations.Npulsars + 1,
                            observations.Npulsars,
                            dictionary_of_known_quantities)



    #First initialise the KF
    UKF_parameters =   {"alpha": 1,  # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
                       "beta":2,       # Beta incorporates prior knowledge of the distribution of x. For Guassians, beta=2 is optimal
                       "kappa":1000.0,       # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
                       }

    KF = UnscentedKalmanFilter(
                            observations=observations,
                            model = model,
                            UKF_settings=UKF_parameters
                            )




    # Then run it for the optimal parameters set of parameters 
    parameters = {
                  "dec_gw":  cfg["GW_parameters"]["dec_GW"],
                  "ra_gw":   cfg["GW_parameters"]["ra_GW"],
                  "psi_gw":  cfg["GW_parameters"]["psi_GW"],
                  "iota_gw": cfg["GW_parameters"]["iota"],
                  "phi0":    cfg["GW_parameters"]["phase_normalisation"],
                  "omega":   observations.omega_GW,
                  "Agw":     observations.Agw, 
                 "gamma":   observations.spindown_gamma[0],
                  "n":       observations.spindown_n[0],
                 }



    model_likelihood = KF.ll_on_data(parameters,"1.0")
    null_likelihood = KF.ll_on_data(parameters,"null")


    print("model likelihood", model_likelihood)
    print("null likelihood", null_likelihood)

    bayes_factor = model_likelihood - null_likelihood
    print("Bayes factor is:",bayes_factor)




if __name__=="__main__":


    import multiprocessing
    multiprocessing.set_start_method("fork") #These lines are needed on macOS since the default start method is "spawn" which doesn't work well with Bilby


    
    np.random.seed(6)


    dt   = 7 #days
    Tend = 10 #years    
    t    = np.arange(0.0,Tend*365*24*3600,dt*24*3600) #time runs from 0 to Tend, with intervals dt 

    Npulsars = 39
    pulsar_parameters = {
         "f_psr":            np.random.uniform(low=50, high=500, size=(Npulsars,)),    
         "dec_psr":          np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(Npulsars,)),  # sampling interval in days
         "ra_psr":           np.random.uniform(low=0.0, high=2*np.pi, size=(Npulsars,)),      # number of pulsars in PTA,
         "pulsar_distances": np.ones(Npulsars),                                               # every pulsar is 1kpc away
         "spindown_gamma":   np.full((Npulsars,),1e-20),
         "spindown_n":       np.full((Npulsars,),3),
         "generate_uniform_pulsars": False  #If true dec_psr and ra_psr are ignored. Generates N pulsars evenly spaced.
          },


    GW_parameters= {
         "omega_GW" :           1e-7,
         "phase_normalisation": 0.20,  
         "psi_GW":              np.random.uniform(low=0.0, high=np.pi*2),
         "iota":                np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "dec_GW":              np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "ra_GW":               np.random.uniform(low=0.0, high=2*np.pi),
         "h0":                  2.5e-8
         
        #  "m1":                  4e12,
        #  "m2":                  3e12,
        #  "Dl":                  1 #Gpc
          },


    noise_parameters = {
                         "process_noise": np.ones(Npulsars)*1e-10,      
                         "measurement_noise": 0.0  # standard deviation gaussian measurement noise.
                       },
   
    
    run(pulsar_parameters[0],GW_parameters[0],noise_parameters[0],t) # [0] unpacks tuple to dict. Why are they tuples?
 


















    #     pulsar_parameters = {
    #      "f_psr":            np.linspace(50, 500, Npulsars),    
    #      "dec_psr":          np.linspace(-np.pi/2, np.pi/2, Npulsars),  # sampling interval in days
    #      "ra_psr":           np.linspace(0.0, 2*np.pi, Npulsars),      # number of pulsars in PTA,
    #      "pulsar_distances": np.ones(Npulsars),                                               # every pulsar is 1kpc away
    #      "spindown_gamma":   np.full((Npulsars,),1e-20),
    #      "spindown_n":       np.full((Npulsars,),3),
    #      "generate_uniform_pulsars": True  #If true dec_psr and ra_psr are ignored. Generates N pulsars evenly spaced.
    #       },


    # GW_parameters= {
    #      "omega_GW" :           1e-7,
    #      "phase_normalisation": 0.20,  
    #      "psi_GW":              1.0,#np.random.uniform(low=0.0, high=np.pi*2),
    #      "iota":                1.0,#np.random.uniform(low=-np.pi/2, high=np.pi/2),
    #      "dec_GW":              1.0,#np.random.uniform(low=-np.pi/2, high=np.pi/2),
    #      "ra_GW":               1.0,#np.random.uniform(low=0.0, high=2*np.pi),
    #      "m1":                  4e12,
    #      "m2":                  3e12,
    #      "Dl":                  1 #Gpc
    #       },


    # noise_parameters = {
    #                      "process_noise": np.ones(Npulsars)*0.0,# 1e-10,      
    #                      "measurement_noise": 0.0  # standard deviation gaussian measurement noise.
    #                    },
   