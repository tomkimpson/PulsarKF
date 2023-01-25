import bilby
import sys
import numpy as np 
import uuid


class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,model,parameters):

        super().__init__(parameters=parameters)
        self.model = model
        
    def log_likelihood(self):
        ll = self.model.ll_on_data(self.parameters,1.0)
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
    flabel = str(uuid.uuid4())[:8] # a random ID. NOT collisionless
    out_path = f"../results/{flabel}/"
    result = bilby.run_sampler(likelihood, priors, label = f'{flabel}',outdir=out_path,
                            sampler ='dynesty',check_point_plot=False,
                            sample='rwalk', walks=10, npoints=100,
                            npool=4,plot=True,resume=False,dlogz=10)

    return result,out_path