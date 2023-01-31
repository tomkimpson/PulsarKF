
"""Model config in json format"""
import numpy as np
Npulsars = 100
#np.random.seed(6)
canonical = {

    "timing_parameters": {
         "T_years" : 10,        # how long to integrate for in years
         "dt_days": 7,          # sampling interval in days
         "Npulsars":Npulsars    # number of pulsars in PTA
          },

    "pulsar_parameters": {
         "f_psr" :  np.full((Npulsars,),100), #np.random.uniform(low=50, high=500, size=(Npulsars,)),    
         "pulsar_distribution": "uniform", #one of random/uniform/orthogonal
         #"dec_psr": np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(Npulsars,)),  #sampling interval in days
         #"ra_psr":  np.random.uniform(low=0.0, high=2*np.pi, size=(Npulsars,)),   #number of pulsars in PTA,
         "pulsar_distances": np.ones(Npulsars), #every pulsar is 1kpc away
         "spindown_gamma": np.full((Npulsars,),1e-20),
         "spindown_n": np.full((Npulsars,),3)
         #"generate_uniform_pulsars": True
          },

    "GW_parameters": {
         "omega_GW" : 1e-7,
         "phase_normalisation": 0.20,  
         "psi_GW":2.5,#np.random.uniform(low=0.0, high=np.pi*2),
         "iota":0.0,#np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "dec_GW":0.0,#np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "ra_GW":1.0,#np.random.uniform(low=0.0, high=2*np.pi),
         "h0": 2.5e-14
         #"m1":4e12,
         #"m2":3e12,
         #"Dl":1e-6 #Gpc
          },


    "noise_parameters":{#"process_noise": np.random.uniform(low=0.001, high=0.002, size=(Npulsars,)),  
                         "process_noise": np.ones(Npulsars)*0.0,#1e-12,      
                         "measurement_noise": 1e-8 # standard deviation gaussian measurement noise.
            },


    "UKF_parameters": {"alpha": 1,  # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
                       "beta":2,       # Beta incorporates prior knowledge of the distribution of x. For Guassians, beta=2 is optimal
                       "kappa":100.0,       # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
                       "measurement_model": "1" # anything or "null". null selects the null measurement model
            }
    
    }
