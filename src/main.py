from UKF import UnscentedKalmanFilter
from create_synthetic_data import PulsarFrequencyObservations
from model import MelatosPTAModel
from bayesian_inference import BilbySampler

from configs.config import canonical as cfg
import numpy as np
import bilby 
import os 




import sys
if __name__=="__main__":


    import multiprocessing
    multiprocessing.set_start_method("fork") #These lines are needed on macOS since the default start method is "spawn" which doesn't work well with Bilby


    #First, let's create some synthetic data.
    dt   = cfg["timing_parameters"]["dt_days"] 
    Tend = cfg["timing_parameters"]["T_years"]     
    t    = np.arange(0.0,Tend*365*24*3600,dt*24*3600) #time runs from 0 to Tend, with intervals dt 

    observations = PulsarFrequencyObservations(t)              # initialise the class, all observations have same times
    observations.create_observations(cfg["pulsar_parameters"],
                                     cfg["GW_parameters"],
                                     cfg["noise_parameters"])  # generate the observations. You can also plot this as e.g. observations.plot_observations(psr_index=2,KF_predictions = None) 
 
    


   
    #Now initialise the state-space model to be used with the UKF
    dictionary_of_known_quantities = {"dec_psr":cfg["pulsar_parameters"]["dec_psr"],
                                      "ra_psr":cfg["pulsar_parameters"]["ra_psr"],
                                      "pulsar_distances":cfg["pulsar_parameters"]["pulsar_distances"]}
    model = MelatosPTAModel(observations.Npulsars + 1,
                            observations.Npulsars,
                            dictionary_of_known_quantities)








    #Now let's run the UKF on this data

    #First initialise the KF
    KF = UnscentedKalmanFilter(
                            observations=observations,
                            model = model,
                            UKF_settings=cfg["UKF_parameters"]
                            )

    #Then run it for a particular set of parameters 
    parameters = {"omega":   observations.omega_GW/1,
                  "gamma":   observations.spindown_gamma[0],
                  "n":       observations.spindown_n[0],
                  "dec_gw":  cfg["GW_parameters"]["dec_GW"],
                  "ra_gw":   cfg["GW_parameters"]["ra_GW"],
                  "psi_gw":  cfg["GW_parameters"]["psi_GW"],
                  "Agw":     observations.Agw,
                  "iota_gw": cfg["GW_parameters"]["iota"],
                  "phi0":    cfg["GW_parameters"]["phase_normalisation"]
                 }



    import time 
    t1 = time.time()
    import cProfile
    KF.ll_on_data(parameters,"1.0")
    #cProfile.run('KF.ll_on_data(parameters,"1.0")',sort="cumtime")
    #likelihod = KF.ll_on_data(parameters,"1.0")
    t2 = time.time()
    print("Time taken:", t2-t1)
    
    #print(parameters["omega"]/observations.omega_GW, likelihod)
    observations.plot_observations(psr_index=4,KF_predictions = KF.IO_array) #Can plot this


    # xx = []
    # yy = []
    # for omega in np.logspace(-8,-6,20):

    #         #First initialise the KF
    #         KF = UnscentedKalmanFilter(
    #                                 observations=observations,
    #                                 model = model,
    #                                 UKF_settings=cfg["UKF_parameters"]
    #                                 )


    #         parameters = {"omega":   omega,
    #               "gamma":   observations.spindown_gamma[0],
    #               "n":       observations.spindown_n[0],
    #               "dec_gw":  cfg["GW_parameters"]["dec_GW"],
    #               "ra_gw":   cfg["GW_parameters"]["ra_GW"],
    #               "psi_gw":  cfg["GW_parameters"]["psi_GW"],
    #               "Agw":     observations.Agw,
    #               "iota_gw": cfg["GW_parameters"]["iota"],
    #               "phi0":    cfg["GW_parameters"]["phase_normalisation"]
    #              }
    #         likelihod = KF.ll_on_data(parameters,1.0)
    #         print(parameters["omega"]/observations.omega_GW, likelihod)
    #         xx.extend([omega])
    #         yy.extend([likelihod])

    # import matplotlib.pyplot as plt
    # plt.plot(xx,yy)
    # plt.xscale('log')
    # plt.show()

    #observations.plot_observations(psr_index=2,KF_predictions = KF.IO_array) #Can plot this



    #A BILBY RUN


    # priors = {  "omega":   bilby.core.prior.LogUniform(1e-8, 1e-5, 'omega'),
    #             "gamma":   observations.spindown_gamma[0],
    #             "n":       float(observations.spindown_n[0]),
    #             "dec_gw":  cfg["GW_parameters"]["dec_GW"],
    #             "ra_gw":   cfg["GW_parameters"]["ra_GW"],
    #             "psi_gw":  cfg["GW_parameters"]["psi_GW"],
    #             "Agw":     observations.Agw,
    #             "iota_gw": cfg["GW_parameters"]["iota"],
    #             "phi0":    cfg["GW_parameters"]["phase_normalisation"]
    #             }


    # result,outpath = BilbySampler(KF,priors)
 









### SCRATCH SPACE



