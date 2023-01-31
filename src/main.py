from UKF import UnscentedKalmanFilter
from create_synthetic_data import PulsarFrequencyObservations
from model import MelatosPTAModel
from bayesian_inference import BilbySampler

from configs.config import canonical as cfg
from configs.config import NF
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
    t    = np.arange(0.0,Tend*365*24*3600,dt*24*3600,dtype=NF) #time runs from 0 to Tend, with intervals dt 

    observations = PulsarFrequencyObservations(t)              # initialise the class, all observations have same times
    observations.create_observations(cfg["pulsar_parameters"],
                                     cfg["GW_parameters"],
                                     cfg["noise_parameters"])  # generate the observations. You can also plot this as e.g. observations.plot_observations(psr_index=2,KF_predictions = None) 

    


    observations.plot_observations(psr_index=2,KF_predictions = None) #Can plot this


    sys.exit()



    #Now initialise the state-space model to be used with the UKF
    dictionary_of_known_quantities = {"pulsar_directions": observations.q,
                                    "pulsar_distances":cfg["pulsar_parameters"]["pulsar_distances"],
                                    "measurement_noise":cfg["noise_parameters"]["measurement_noise"],
                                    }
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
    parameters = {"omega":   observations.omega_GW,
                "gamma":   observations.spindown_gamma[0],
                "n":       observations.spindown_n[0],
                "dec_gw":  cfg["GW_parameters"]["dec_GW"],
                "ra_gw":   cfg["GW_parameters"]["ra_GW"],
                "psi_gw":  cfg["GW_parameters"]["psi_GW"],
                "Agw":     observations.Agw,
                "iota_gw": cfg["GW_parameters"]["iota"],
                "phi0":    cfg["GW_parameters"]["phase_normalisation"]
                }




    

    model_likelihood = KF.ll_on_data(parameters,"1.0")
    #observations.plot_observations(psr_index=2,KF_predictions = KF.IO_array) #Can plot this

    null_likelihood = KF.ll_on_data(parameters,"null")
    #observations.plot_observations(psr_index=2,KF_predictions = KF.IO_array) #Can plot this


    print("model likelihood", model_likelihood)
    print("null likelihood", null_likelihood)


    bayes_factor = model_likelihood - null_likelihood


    print("Likelihood ratio:", bayes_factor)


    #observations.plot_observations(psr_index=2,KF_predictions = KF.IO_array) #Can plot this




# #     import time 
# #     t1 = time.time()
# #     import cProfile
    #likelihood = KF.ll_on_data(parameters,"1.0")
# #     #cProfile.run('KF.ll_on_data(parameters,"1.0")',sort="cumtime")
# #     #likelihod = KF.ll_on_data(parameters,"1.0")
# #     t2 = time.time()
    #print("likelihood:",likelihood)
    
# #     #print(parameters["omega"]/observations.omega_GW, likelihod)
#     observations.plot_observations(psr_index=4,KF_predictions = KF.IO_array) #Can plot this


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


    # priors = {  "omega":   bilby.core.prior.LogUniform(1e-9, 1e-5, 'omega'),
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
 






### Plot N random orthogonal pulsars

    # GW_direction = observations.gw_dir
    

    # N = 100


    # pulsars = np.zeros((N,3))

    # for i in range(N):
    #     x = np.random.randn(3)  # take a random vector
    #     x -= x.dot(GW_direction) * GW_direction       # make it orthogonal to k
    #     x /= np.linalg.norm(x)  # normalize it

    #     pulsars[i,:] = x 



    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')



    # ax.scatter(0, 0, 0,c='r')
    # ax.scatter(GW_direction[0],GW_direction[1],GW_direction[2],c='k')


    # for i in range(N):
    #     ax.plot([0,pulsars[i,0]],[0,pulsars[i,1]],[0,pulsars[i,2]])

    #     print(np.dot(GW_direction,pulsars[i,:]))
    #     #ax.scatter(pulsars[i,0],pulsars[i,1],pulsars[i,2],c='C0')
    # #ax.scatter(y[0],y[1],y[2],c='C0')


    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-1,1)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # plt.show()


    # sys.exit()