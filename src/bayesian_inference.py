import bilby
import sys
import numpy as np 


class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,model,parameters):

        super().__init__(parameters=parameters)
        self.model = model
        
    def log_likelihood(self):
        ll = self.model.ll_on_data(self.parameters)
        return ll




def BilbySampler(KalmanFilter,priors_on_parameters):
   
    #Initialise a Bilby Likelihood and priors
    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    for key,value in priors_on_parameters.items():
        init_parameters[key] = None
        priors[key] = value


    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    flabel = "TEST_"
    out_path = "../results/"
    result = bilby.run_sampler(likelihood, priors, label = f'{flabel}',outdir=out_path,
                            sampler ='dynesty',check_point_plot=False,
                            sample='rwalk', walks=10, npoints=100,
                            npool=4,plot=True,resume=False)

    return result