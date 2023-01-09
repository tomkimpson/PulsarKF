

import numpy as np 
from scipy.linalg import sqrtm as matrix_sqrt

class UnscentedKalmanFilter:

    """
    A class to implement the Unscented Kalman Filter as described in e.g. https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    
        n_states: dimension (number of states) of your random variable x

        observations: the actual data to run the UKF on. Dimension of n_observations x n_times
    
    
    """


    #Constants 


    def __init__(self,n_states,observations):

        self.L = n_states 
        self.observations = observations
        
        
        #Initialise some constants of the class
        self.alpha = 7e-4 # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
        self.beta  = 2    # Beta incorporates prior knowledge of th distribution of x. For Guassians, beta=2 is optimal
        self.kappa = 0    # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
        
        

        
        

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
        self.gamma = np.sqrt(self.nstates + lamda)

    def _calculate_sigma_points(self):

        """
        Internal function

        Calculate the sigma vectors

        Updates self.chi
        """

        #Initialise the sigma vector
        self.chi = np.zeros((2*self.L + 1,1))

        #The 0th element is just the mean
        self.chi[0] = self.x

        #Then iterate over the remaining elements
        P_sqrt = matrix_sqrt(self.P) 
        for i in range(1,self.L): self.chi[i] = self.x +(self.gamma * P_sqrt[i,:])




        

    def ll_on_data(self):

        """
        External function




        """


        #Initialise x and P
        self.x = np.ones((self.L,1)) # a column vector, length L
        self.P = np.eye(self.L)      # a square matrix, dim(L x L)   

        #Determine the weights for the UKF
        self._calculate_weights()






    #def _predict(self):


    #def _update(self)
