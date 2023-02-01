
"""Model config in json format"""
import numpy as np
Npulsars = 45
NF = np.float128
#np.random.seed(6)
canonical = {

    "timing_parameters": {
         "T_years" : 10,        # how long to integrate for in years
         "dt_days": 7,          # sampling interval in days
         "Npulsars":Npulsars    # number of pulsars in PTA
          },

    "pulsar_parameters": {
         "f_psr" :           np.full((Npulsars,),100,dtype=NF), #np.random.uniform(low=50, high=500, size=(Npulsars,)),    
         "pulsar_distances": np.ones(Npulsars,dtype=NF), #every pulsar is 1kpc away
         "spindown_gamma":   np.full((Npulsars,),1e-20,dtype=NF),
         "spindown_n":       np.full((Npulsars,),3,dtype=NF),
         "pulsar_distribution": "NANOGrav", #one of random/uniform/orthogonal/NANOGrav. If NANOGrav all pulsar parameters are overriden with the true parameters of the NANOGrav pulsars
          },

    "GW_parameters": {
         "omega_GW":            NF(1e-7),
         "phase_normalisation": NF(0.20),  
         "psi_GW":              NF(2.5),
         "iota":                NF(0.0),
         "dec_GW":              NF(0.0),
         "ra_GW":               NF(1.0),
         "h0":                  NF(1e-12)
          },


    "noise_parameters":{
                         "process_noise": NF(2e-16),    #standard deviation of process noise      
                         "measurement_noise": NF(0.0) #standard deviation gaussian measurement noise.
            },


    "UKF_parameters": {"alpha": 1,         # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
                       "beta":  int(2),         # Beta incorporates prior knowledge of the distribution of x. For Guassians, beta=2 is optimal
                       "kappa": 100,       # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
                       "measurement_model": "1" # anything or "null". null selects the null measurement model
            }
    
    }
