
"""Model config in json format"""
import numpy as np
Npulsars = 5
np.random.seed(0)
canonical = {

    "timing_parameters": {
         "T_years" : 10,        #how long to integrate for in years
         "dt_days": 7,         #sampling interval in days
         "Npulsars":Npulsars   #number of pulsars in PTA
          },

    "pulsar_parameters": {
         "f_psr" :  np.random.uniform(low=50, high=500, size=(Npulsars,)),    
         "dec_psr": np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(Npulsars,)),  #sampling interval in days
         "ra_psr":  np.random.uniform(low=0.0, high=2*np.pi, size=(Npulsars,)),   #number of pulsars in PTA,
         "pulsar_distances": np.ones(Npulsars), #every pulsar is 1kpc away
         "spindown_gamma": np.full((Npulsars,),1e-20),
         "spindown_n": np.full((Npulsars,),3),
          },

    "GW_parameters": {
         "omega_GW" : 1e-8,
         "phase_normalisation": 0.20,  
         "psi_GW":np.random.uniform(low=0.0, high=np.pi*2),
         "iota":np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "dec_GW":np.random.uniform(low=-np.pi/2, high=np.pi/2),
         "ra_GW":np.random.uniform(low=0.0, high=2*np.pi),
         "m1":4e9,
         "m2":3e9,
         "Dl":0.010 #Gpc
          },


    "noise_parameters":{"process_noise": np.random.uniform(low=0.001, high=0.002, size=(Npulsars,)),        
                        "measurement_noise": 1e-15    # standard deviation gaussian measurement noise.
            }



    
    }
