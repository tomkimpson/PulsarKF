

import numpy as np 
from scipy.linalg import sqrtm as matrix_sqrt
from scipy.linalg import qr #as qr_decomposition
import scipy.linalg as la
import sys
from scipy.stats import multivariate_normal
from configs.config import NF

class UnscentedKalmanFilter:

    """
    A class to implement the Unscented Kalman Filter as described in e.g. https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    
    It takes two arguments 
        

        observations: A class which holds the the actual data to run the UKF on. 
    
        model: The definition of the Kalman machinery: transition functions, measurement functions etc.
    
    """

    def __init__(self,observations,model,UKF_settings):


        
        self.observations = observations.observations # The actual noisy data recorded by an observer

        self.dt = observations.dt # Currently assumes all observations are equally spaced by dt, so dt is not indexed
        self.L = model.dims_x     # dimension of hidden states
        
        
        #Initialise some constants of the class

        self.alpha = UKF_settings["alpha"]
        self.beta  = UKF_settings["beta"]
        self.kappa = UKF_settings["kappa"]


        
        #Initialise the functions defined by the model
        self.initialise_model = model.initialize_global_quantities
        self.Q_function       = model.Q_function
        self.F_function       = model.F_function
        self.R                = model.R_function()

        self.H_measure = model.H_function
        self.H_noise   = model.null_measurement_function


        #Determine the weights for the UKF. This only needs to be done once.
        self._calculate_weights()

        

    def _predict(self,sigma_points):

        #("predict")
        #See Eq 17/18 from Wan/Van     
        x_predicted = np.dot(self.Wm, sigma_points)#tmp = self.Wm.reshape(13,1), equivalent to np.sum(sigma_points*tmp,axis=0)
        
       
       
        #This form combining dot products with diag of the weights array is taken from: https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/UKF.py#L284
        #Todo: caclulate diag(weights) ONCE
        y = sigma_points - x_predicted[np.newaxis, :]



        




        P_predicted = np.dot(y.T,np.dot(np.diag(self.Wc),  y))
        P_predicted = 0.5*(P_predicted + P_predicted.T) #+ 1e-16*np.eye(len(P_predicted)) # Enforce symmetry of the covariance matrix
        
        # i = 1
        # j = 2
        # for i in range(len(P_predicted)):
        #     for j in range(len(P_predicted)):
        #         print(i,j,P_predicted[i,j],P_predicted[j,i])
        # sys.exit()

        return x_predicted, P_predicted,y # return y as we will use it later when calculating cross correlation



    def _predict2(self,sigma_points):


        #See Eq 17/18 from Wan/Van     
        x_predicted = np.dot(self.Wm, sigma_points)#tmp = self.Wm.reshape(13,1), equivalent to np.sum(sigma_points*tmp,axis=0)
        


        return x_predicted







    def _update(self,observation):

        """
        Measurement update equations
        """


        #Get the cross correlation matrix
        Pxy = np.dot(self.delta_y.T, np.dot(np.diag(self.Wc),self.delta_x))

        

        #Get the Kalman gain
        Q,R = np.linalg.qr(np.float64(self.P_yy)) #Don't confuse this Q,R decomposition with self.Q and self.R for the noises. self.P_yy has to be downsampled for use with LinearAlgebra libraries  
        

        
        Qb = np.dot(Q.T,Pxy)
        K = np.linalg.solve(R,np.float64(Qb)).T #The transpose makes the shape appropriate for a dot product with the innovation below
        #K_full_definition = Pxy.T @ np.linalg.inv(self.P_yy) #transpose just required for correct shapes
       
       
        
        #Update the state and covariance estimates
        innovation =  observation - self.y_predicted 
    

        self.x = self.x_predicted + K @ innovation
        self.P = self.P_xx - K @ self.P_yy @ K.T
        
        #print("Updated uncertainty")
        #sys.exit()
    
        #Also return the likelihood    
        likelihood = -0.5*(np.linalg.slogdet(np.float64(self.P_yy))[1] + innovation.T @ np.linalg.solve(np.float64(self.P_yy), np.float64(innovation))+ len(observation) * np.log(2*np.pi))
        self.ll += likelihood
        
    def _calculate_weights(self):

        """
        Internal function

        Calculate the weights of the UKF.

        Updates self.Wm, self.Wc
        """

          
        lambda_ = self.alpha**2 *(self.L+self.kappa) - self.L # scaling parameter used in calculating the weights

        #Preallocating arrays then filling to make dimensions explicit.
        #Verbose, but clear. Maybe just use np.full()...
        self.Wm = np.zeros(2*self.L+1,dtype=NF)  
        self.Wc = np.zeros(2*self.L+1,dtype=NF)



        #Fill Wm
        self.Wm[0] = lambda_  / (self.L + lambda_ )
        for i in range(1,len(self.Wm)):
            self.Wm[i] = 1/(2*(self.L+lambda_ ))

        #Fill Wc
        self.Wc[0] = lambda_  / (self.L + lambda_ ) + (1 - self.alpha**2 + self.beta)
        for i in range(1,len(self.Wc)):
            self.Wc[i] = 1/(2*(self.L+lambda_ ))


        #Also define....
        self.gamma = np.sqrt(self.L + lambda_,dtype=NF)

    def _calculate_sigma_vectors(self, x,P): 

        """
        Internal function

        Calculate the sigma vectors for a given state `x` and covariance `P`

        Updates self.chi

        See Eq. 15 from Wav/Van
        """

        #print("calculate sigma vectors")
        #Check if the P matrix can be sqrted
        #epsilon = sys.float_info.epsilon
        #P_check = 0.5*(P + P.T) + epsilon*np.eye(len(x)) #uncomment these two lines, and comment out P_check=P, if P sqrts are causing issues.
        P_check = P
        
        P_sqrt = la.cholesky(P_check, check_finite=True)  #Cholesky is much faster than scipy.linalg.sqrtm. Note that this is float64 not float128
        

        

       
        M = self.gamma*P_sqrt
       
        self.chi = np.zeros((2*self.L + 1,self.L),dtype=NF)

       
        self.chi[0,:] = x

        self.chi[1:self.L+1,:] = x + M
        self.chi[self.L+1:2*self.L+1+1,:] = x - M




    def _calculate_sigma_vectors2(self, x,S): 

        """
        Internal function

        Calculate the sigma vectors for a given state `x` and covariance `P`

        Updates self.chi

        See Eq. 15 from Wav/Van
        """

       
        M = self.gamma*S
       
        self.chi = np.zeros((2*self.L + 1,self.L),dtype=NF)

       
        self.chi[0,:] = x

        self.chi[1:self.L+1,:] = x + M
        self.chi[self.L+1:2*self.L+1+1,:] = x - M




       
    def ll_on_data(self,parameters,measurement_model):

        """
        External function
        """

        if measurement_model == "null":
            self.H_function       = self.H_noise
            print("Using the null measurement")
        else:
            self.H_function       = self.H_measure


        print("Welcome to LL on data")

        #Initialize the model
        self.initialise_model(parameters)

        #Initialise the array that will store the results
        self.IO_array = np.zeros((len(self.observations),self.L),dtype=NF)

        #Initialise x and P
        self.x    = np.ones(self.L,dtype=NF) # a column vector, length L
        self.x[0] = parameters["phi0"]       # this is the only place this parameter comes in

        self.x[1:self.L] = self.observations[0,:] # guess that the intrinsic frequency is the same as the measured frequancy

        self.P = np.eye(self.L,dtype=NF)*1e-3#*100 # a square matrix, dim(L x L). #How to initialise?
        self.P[0,0] = 9e-16

 

        self.S = la.cholesky(self.P, check_finite=True)  #Cholesky is much faster than scipy.linalg.sqrtm. Note that this is float64 not float128
        #print(self.S)
        #sys.exit()





        #Initialise the likelihood
        self.ll = 0

        i = 0 # a useful counter
        for observation in self.observations:
            #print(f"Observation number {i} ", observation)
        
            # 1. Calculate sigma points, given the state variables 
            self._calculate_sigma_vectors(self.x,self.P) 
            #self._calculate_sigma_vectors2(self.x,self.S) 

           
            
            # 2. Time update
            
            # 2.1 Update the process noise covariance 
            #---at the moment for our problem Q is constant in time, leaving it here anyway
            self.Q = self.Q_function(self.x,self.dt) 

 
            # 2.1 Evolve the sigma points in time
            self.sigma_points_x = self.F_function(self.chi,self.dt)
            
   
            
            # 2.2 Weighted state predictions and covariance
            #print("Get Pxx")
            self.x_predicted, self.P_xx,self.delta_x = self._predict(self.sigma_points_x) 
            #self.x_predicted = self._predict2(self.sigma_points_x) 
            #print(self.P_xx.shape)
            #sys.exit()

            #sys.exit()
            # print(self.x_predicted[np.newaxis, :].shape)
            # print(self.sigma_points_x.shape)

            # Qsqrt = la.cholesky(self.Q, check_finite=True) 
            # part1 = np.sqrt(self.Wc[1]) * (self.sigma_points_x - self.x_predicted[np.newaxis, :])
            # compound_matrix =  np.dot(part1,Qsqrt)
            # print(compound_matrix.shape)
            # blob,self.S_xx = np.linalg.qr(np.float64(compound_matrix))
            # sys.exit()

            # print(compound_matrix.shape)
            # print(self.x_predicted.shape)
            # sys.exit()





            #blob, self.S_xx =np.linalg.qr(np.float64(self.P_yy)) #Don't confuse this Q,R decomposition with self.Q and self.R for the noises. self.P_yy has to be downsampled for use with LinearAlgebra libraries  


            self.P_xx += self.Q
            

            # 2.3 Update the sigma vectors using these new predictions of the state/covariance
            #---i.e. this updates self.chi
            #Todo: Do we need this step? Seems to be slightly conflicting advice in the literature.
            #Joe's nice working code does include this step See also e.g. 
            #self._calculate_sigma_vectors(self.x_predicted,self.P_xx)
            
            # 2.4 Evolve these new sigma vectors according to the measurement function
            #--- note that the dimensions of self.sigma_points_x and  self.sigma_points_y are different!
            self.sigma_points_y = self.H_function(self.chi) # this is 4th step in time update. 
           

            # 2.5 Weighted state predictions and covariance
            self.y_predicted, self.P_yy,self.delta_y = self._predict(self.sigma_points_y) # 


            self.P_yy += self.R
            
            # 3. Measurement update
            self._update(observation) 

            #Do some IO
            self.IO_array[i,:] = self.x
            

            i += 1


            #if i >2:
               # sys.exit("TK EXIT")
        

        return self.ll
            















