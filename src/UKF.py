

import numpy as np 
from scipy.linalg import sqrtm as matrix_sqrt

class UnscentedKalmanFilter:

    """
    A class to implement the Unscented Kalman Filter as described in e.g. https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    
        n_states: dimension (number of states) of your random variable x

        observations: the actual data to run the UKF on. Dimension of n_observations x n_times
    
    
    """


    #Constants 


    def __init__(self,n_states,observations,Q, F,H):

#propagate_sigma_points_function = this is just ODE solution



        self.L = n_states 
        self.observations = observations.observations #The actual noisy data recorded by an observer
        self.dt = observations.dt #assumes all observations are equally spaced by dt
        
        #Initialise some constants of the class
        self.alpha = 7e-4 # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
        self.beta  = 2    # Beta incorporates prior knowledge of th distribution of x. For Guassians, beta=2 is optimal
        self.kappa = 0    # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
        
        

        self.Q_function = Q 
        self.F_function = F
        self.H_function = H
        
    def _predict(self):

        #See Eq 17/18 from Wan/Van


        self.x_predicted = sum(self.Wm* self.sigma_points)

        delta = self.sigma_points - self.x_predicted
        
        print (self.x_predicted.shape)
        print (self.sigma_points.shape)
        print(delta.shape)
        print(self.Wc.shape)
        self.P_predicted = sum(self.Wc * np.outer(delta,delta) ) + self.Q

    def _update(self):

        #see O'leary and Measurement equations section of paper
        return 1

    def _calculate_weights(self):

        """
        Internal function

        Calculate the weights of the UKF.

        Updates self.Wm, self.Wc
        """


        lamda = self.alpha**2 *(self.L+self.kappa) - self.L #scaling parameter used in calculating the weights

        #The weights
        self.Wc = np.full(2*self.L + 1,  1. / (2*(self.L + lamda)))
        self.Wm = self.Wc

        #Overwrite the 0th elements
        self.Wm[0] = lamda / (self.L + lamda)
        self.Wc[0] = lamda / (self.L + lamda) + (1.0 - self.alpha**2 + self.beta)
        self.gamma = np.sqrt(self.L + lamda)

    def _calculate_sigma_vectors(self, x,P): 

        """
        Internal function

        Calculate the sigma vectors for a given state `x` and covariance `P`

        Updates self.chi

        See Eq. 15 from Wav/Van
        """

        #Initialise the sigma vector
        self.chi = np.zeros((2*self.L + 1,self.L))

        #The 0th element is just the mean
        self.chi[0,:] = x#.flatten()

        #Then iterate over the remaining elements
        P_sqrt = matrix_sqrt(P) 
        for i in range(1,self.L): self.chi[i,:]          = x +(self.gamma * P_sqrt[i,:])
        for i in range(self.L+1,2*self.L): self.chi[i,:] = x -(self.gamma * P_sqrt[i - self.L,:])


    

    def ll_on_data(self,parameters):

        """
        External function




        """


        #Initialise x and P
        self.x = np.ones(self.L) # a column vector, length L
        self.P = np.eye(self.L)      # a square matrix, dim(L x L)   

        #Determine the weights for the UKF. This only needs to be done once.
        self._calculate_weights()


        for observation in self.observations:
            #print(observation)
            print("Loading an observation")
            
            print ("Calculating sigma vector")
            self._calculate_sigma_vectors(self.x,self.P) # Calculate sigma points, given the state variables 
            print(self.chi.shape)
            print ("Propagating sigma vector")
            self.sigma_points = self.F_function(self.chi,parameters,self.dt)       # Propagate the sigma points
            print(self.sigma_points.shape)
            
            
            print("Getting Q function")
            self.Q = self.Q_function()                  # Update the process noise covariance 
            print("predict")
            self._predict() # caveat if there are no observations at this time, state = predict NO UPDATE step


            self._calculate_sigma_points(self.x_predicted)
            self.measurement_operator_function() #this is 4th step in time update. 
            self._update() 
            #


            #at t=0, prop sigmas = sigmas already calculated (not compulsoary)

            #o


    #def _predict(self):


    #def _update(self)
