

import numpy as np 
from scipy.linalg import sqrtm as matrix_sqrt
from scipy.linalg import qr #as qr_decomposition
import scipy.linalg as la
import sys
class UnscentedKalmanFilter:

    """
    A class to implement the Unscented Kalman Filter as described in e.g. https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    
    It takes two arguments 
        

        observations: A class which holds the the actual data to run the UKF on. 
    
        model: The definition of the Kalman machinery: transition functions, measurement functions etc.
    
    """

    def __init__(self,observations,model):


        
        self.observations = observations.observations # The actual noisy data recorded by an observer

        self.dt = observations.dt # Currently assumes all observations are equally spaced by dt, so dt is not indexed
        self.L = model.dims_x     # dimension of hidden states
        
        
        #Initialise some constants of the class
        self.alpha = 7e-4 # O'Leary uses 7e-4, Wan/Merwe recommend something generally small. e.g. 1e-3
        self.beta  = 2    # Beta incorporates prior knowledge of th distribution of x. For Guassians, beta=2 is optimal
        self.kappa = 0    # An extra scaling parameter. O'Leary uses 3-n_states, Wan/Merwe recommend =0
        
        

        self.initialise_model = model.initialize_global_quantities
        self.Q_function = model.Q_function
        self.F_function = model.F_function
        self.H_function = model.H_function
        self.R = model.R_function()
        
    def _predict(self,sigma_points):

        #See Eq 17/18 from Wan/Van
        x_predicted  = (sigma_points * self.Wm[:, None]).sum(axis=0) #This is Equation 17 from Wan/Van
        P_predicted = self._calculate_covariance(x_predicted,sigma_points,x_predicted,sigma_points,self.Wc) 

        return x_predicted, P_predicted 

    def _update(self,observation):

        """
        Measurement update equations
        """


        innovation =  observation - self.y_predicted 
        
        Pxy = self._calculate_covariance(self.x_predicted,
                                         self.sigma_points_x,
                                         self.y_predicted,
                                         self.sigma_points_y,
                                         self.Wc) 

       

       
        Q,R = np.linalg.qr(self.P_yy) #Don't confuse this Q,R decomposition with self.Q and self.R for the noises  
        Qb = Pxy @ Q.T
        KalmanGain = np.linalg.solve(R,Qb.T).T



        #NEW METHOD 
        print("KALMAN GAIN")
        Kgain_new =Pxy @ np.linalg.inv(self.P_yy)

        #K3 = Pxy @ np.linalg.inv(R) @ Q.T

        inverse1 = np.linalg.inv(self.P_yy)
        inverse2 = np.linalg.inv(R) @ Q.T
        #print("The inverse of Pyy is:", np.linalg.inv(self.P_yy))
        #print("The QR decomp of Pyy is:", np.linalg.inv(R) @ Q.T)
        print("LOOK HERE")
        print(inverse1- inverse2)
       
        print(KalmanGain - Kgain_new)


        sys.exit()







        
        print ("Are these Kgains equal?", np.array_equal(Kgain_new,KalmanGain))
        #print ("Are these Kgains equal?", Kgain_new == KalmanGain)
        print(Kgain_new == K3)
        print("the difference is:", KalmanGain - Kgain_new)

        #Update the state and covariance
        self.x = self.x_predicted + Kgain_new @ innovation
        #self.P = self.P_xx - np.dot(Kgain_new, np.dot(self.P_yy,Kgain_new.T))
        self.P = self.P_xx - Kgain_new @ self.P_yy @ Kgain_new.T

        #Also return the likelihood
        #likelihood = -0.5*(np.linalg.slogdet(self.P_yy)[1] + innovation.T @ np.linalg.solve(self.P_yy, innovation)+ len(observation) * np.log(2*np.pi))
       

        



        return 1.0 #likelihood

    def _calculate_weights(self):

        """
        Internal function

        Calculate the weights of the UKF.

        Updates self.Wm, self.Wc
        """

          
        lambda_ = self.alpha**2 *(self.L+self.kappa) - self.L #scaling parameter used in calculating the weights

        #Preallocating arrays then filling to make dimensions explicit.
        #Verbose, but clear. Maybe just use np.full()...
        self.Wm = np.zeros(2*self.L+1)  
        self.Wc = np.zeros(2*self.L+1)



        #Fill Wm
        self.Wm[0] = lambda_  / (self.L + lambda_ )
        for i in range(1,len(self.Wm)):
            self.Wm[i] = 1.0/(2*(self.L+lambda_ ))

        #Fill Wc
        self.Wc[0] = lambda_  / (self.L + lambda_ ) + (1.0 - self.alpha**2 + self.beta)
        for i in range(1,len(self.Wc)):
            self.Wc[i] = 1.0/(2*(self.L+lambda_ ))


        #Also define....
        self.gamma = np.sqrt(self.L + lambda_ )

    def _calculate_sigma_vectors(self, x,P): 

        """
        Internal function

        Calculate the sigma vectors for a given state `x` and covariance `P`

        Updates self.chi

        See Eq. 15 from Wav/Van
        """

       
        #Check if the P matrix can be sqrted
        epsilon = 1e-18
        Pos_definite_Check= 0.5*(P + P.T) + epsilon*np.eye(len(x))
        print("Checking for postive definitenes of the P matrix")
        print ("PMatrix to be checked:")
        print(P)
        U = la.cholesky(Pos_definite_Check, check_finite=True) #.T
        #print(U)

        #Initialise the sigma vector
        self.chi = np.zeros((2*self.L + 1,self.L))

        #The 0th element is just the mean
        self.chi[0,:] = x

        #Then iterate over the remaining elements
        #P_sqrt = matrix_sqrt(P) 
        P_sqrt = U
        for i in range(1,self.L+1): 
            self.chi[i,:]          = x +(self.gamma * P_sqrt[i-1,:])
        for i in range(self.L+1,2*self.L+1): 
            self.chi[i,:] = x -(self.gamma * P_sqrt[i-1 - self.L,:])

   

    def _calculate_covariance(self,a,sigma_a,b,sigma_b,weights):

        """
        Given two random vectors `sigma_a`, `sigma_b` (i.e. matrices corresponding to sigma vectors)
        and the associated means - `a` and `b` - calculate cross-covariance matrix. 
        """

        dim_a = len(a)
        dim_b = len(b)
        output = np.zeros((dim_a,dim_b)) 
        
        #print("dimensions:", dim_a, dim_b)
        
        delta_a = sigma_a - a
        delta_b = (sigma_b - b)#.T

        #print("shape of delta a", delta_a.shape)
        #print("shape of delta b", delta_b.shape)


        #Todo: I'm sure there is an efficient way to vectorise this.
        #Leaving explicit for now to make dimensions clear
        for i in range(dim_a):
            for j in range(dim_b):
                scalar = 0.0
                #print(i,j)
                for k in range(2*self.L+1):
                    #print("K summation:", k)
                    scalar += weights[k]*delta_a[k,i]*delta_b[k,j] 
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
        self.x[1:self.L] = 100.0 # Give a more reasonable guess for the psr frequencies
        self.P = np.eye(self.L)#*1e-15*1e-5      # a square matrix, dim(L x L)   

        #Determine the weights for the UKF. This only needs to be done once.
        self._calculate_weights()


        for observation in self.observations:
            print("Observation:", observation)
        
            # 1. Calculate sigma points, given the state variables 
            print("CALCULATING SIGMA VECTORS")
            self._calculate_sigma_vectors(self.x,self.P) 
           
            
            # 2. Time update
            
            # 2.1 Update the process noise covariance 
            #---at the moment for our problem Q is constant in time, leaving it here anyway
            self.Q = self.Q_function(self.dt) 

            # 2.1 Evolve the sigma points in time
            self.sigma_points_x = self.F_function(self.chi,parameters,self.dt)       
            
            # 2.2 Weighted state predictions and covariance
            #print("predict step 1")
            self.x_predicted, self.P_xx = self._predict(self.sigma_points_x) 
            self.P_xx += self.Q

            # 2.3 Update the sigma vectors using these new predictions of the state/covariance
            #---i.e. this updates self.chi
            #self._calculate_sigma_vectors(self.x_predicted,self.P_xx)
            
            # 2.4 Evolve these new sigma vectors according to the measurement function
            #--- note that the dimensions of self.sigma_points_x and  self.sigma_points_y are different!
            self.sigma_points_y = self.H_function(self.chi,parameters) # this is 4th step in time update. 
            
            # 2.5 Weighted state predictions and covariance
            #print("predict step 2")
            self.y_predicted, self.P_yy = self._predict(self.sigma_points_y) # 
            self.P_yy += self.R
            # 3. Measurement update
            self._update(observation) 


            #sys.exit()
            













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