

import numpy as np 
from scipy.linalg import sqrtm as matrix_sqrt
import sys
class UnscentedKalmanFilter:

    """
    A class to implement the Unscented Kalman Filter as described in e.g. https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    
        n_states: dimension (number of states) of your random variable x

        observations: the actual data to run the UKF on. Dimension of n_observations x n_times
    
    
    """




    def __init__(self,n_states,observations,model):


        self.L = n_states 
        self.observations = observations.observations #The actual noisy data recorded by an observer
        self.dt = observations.dt #assumes all observations are equally spaced by dt
        
        #Initialise some constants of the class
        self.alpha = 7e-4 # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
        self.beta  = 2    # Beta incorporates prior knowledge of th distribution of x. For Guassians, beta=2 is optimal
        self.kappa = 0    # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
        
        

        self.initialise_model = model.initialize_global_quantities
        self.Q_function = model.Q_function
        self.F_function = model.F_function
        self.H_function = model.H_function
        
    def _predict(self,sigma_points):

        #See Eq 17/18 from Wan/Van

        
        x_predicted  = (sigma_points * self.Wm[:, None]).sum(axis=0) #This is Equation 17 from Wan/Van
        
        P_predicted = self._calculate_covariance(x_predicted,sigma_points,x_predicted,sigma_points,self.Wc)


        return x_predicted, P_predicted + self.Q



    def _update(self,observation):

        #see O'leary and Measurement equations section of paper
        print ("THIS IS THTE UPDATE STEP")


        innovation = self.y_predicted - observation
        
        Pxy = self._calculate_covariance(self.x_predicted,
                                         self.sigma_points_x,
                                         self.y_predicted,
                                         self.sigma_points_y,
                                         self.Wc) 
        print(1)
        sys.exit()

        



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


    def _calculate_covariance(self,a,a_covar,b,b_covar,weights):

        """
        Given two random vectors `a_covar`, `b_covar` (i.e. matrices corresponding to sigma vectors)
        and the associated means  `a` and `b`, calculate cross-covariance matrix 
        """

        dim_a = len(a)
        dim_b = len(b)
        output = np.zeros((dim_a,dim_b)) 
        
        
        delta_a = a_covar - a
        delta_b = b_covar - b

        #Todo: I'm sure there is an efficient way to vectorise this.
        #Leaving explicit for now to make dimensions clear
        for i in range(dim_a):
            for j in range(dim_b):
                scalar = 0.0
                for k in range(2*self.L+1):
                    scalar += weights[k]*delta_a[k,i]*delta_b[k,j].T 
                output[i,j] = scalar

        return output


    def ll_on_data(self,parameters):

        """
        External function
        """

        #Initialize the model
        self.initialise_model(parameters)


        #Initialise x and P
        self.x = np.ones(self.L) # a column vector, length L
        self.P = np.eye(self.L)      # a square matrix, dim(L x L)   

        #Determine the weights for the UKF. This only needs to be done once.
        self._calculate_weights()


        for observation in self.observations:


            # 1. Calculate sigma points, given the state variables 
            self._calculate_sigma_vectors(self.x,self.P) 
            
            #2. Time update
            
            #2.1 Update the process noise covariance 
            self.Q = self.Q_function() 


            #2.1 Evolve the sigma points in time
            self.sigma_points_x = self.F_function(self.chi,parameters,self.dt)       # Propagate the sigma points
            
            #2.2 Weighted state predictions and covariance
            self.x_predicted, self.P_xx = self._predict(self.sigma_points_x) # For the general case there is a caveat if there are no observations at this time, state = predict NO UPDATE step

            #print("x pred:", self.x_predicted)
            #print("P pred:", self.P_xx)
            #2.3 Update the sigma vectors using these new predictions of the state/covariance
            #---i.e. this updates self.chi
            self._calculate_sigma_vectors(self.x_predicted,self.P_xx)
            
            #2.4 Evolve these new sigma vectors according to the measurement function
            #--- note that the dimensions of self.sigma_points_x and  self.sigma_points_y are different!
            print("----------------measurement----------------------")
            self.sigma_points_y = self.H_function(self.chi,parameters) # this is 4th step in time update. 
            
            #2.5 Weighted state predictions and covariance
            self.y_predicted, self.P_yy = self._predict(self.sigma_points_y) # 
            

            #3. Measurement update
            self._update(observation) 
            #













###--------------Scratch space

        #This is a convoluted, but explicit way of doing the summation.
        #Todo: check dimensionality with Joe and make more concise 
        # self.P_predicted = np.zeros_like(self.P)
        # for i in range(self.L):
            
        #     delta = sigma_points - x_predicted[i,None]
            
        #     tmp_array = np.zeros((2*self.L+1, self.L))
        #     for j in range(2*self.L+1):
        #         tmp_array[j,:] =self.Wc[j]*delta[j,:]*delta[j,:].T 
              

        #     self.P_predicted[i,:] = tmp_array.sum(axis=0)
            
        # #todo: where does the + Q come from? Can't see if defined in Wan/Van.
        # #self.P_predicted = _calculate_covariance(self,a,a_covar,b,b_covar,weights) + self.Q

        # print("Original method is:", self.P_predicted)