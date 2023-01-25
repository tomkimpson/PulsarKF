import numpy as np 
from scipy.integrate import odeint
from universal_constants import *
import sys 

from GW import * 

#This whole thing should really be a child class that inherits a parent





class MelatosPTAModel:


    def __init__(self,dims_x,dims_z,dictionary_of_known_quantities):


        self.dims_x = dims_x
        self.dims_z = dims_z
        


        dec_psr = dictionary_of_known_quantities["dec_psr"]
        ra_psr  = dictionary_of_known_quantities["ra_psr"]
        self.pulsar_distances = dictionary_of_known_quantities["pulsar_distances"] * 1e3 * pc #from kpc to m
        
        self.q = pulsar_directions(np.pi/2.0 - dec_psr,ra_psr)           # Get the direction vector of the pulsar


        self.q_ij = np.zeros((len(self.q),9))
        for i in range(len(self.q)):
            row = self.q[i,:].reshape(3,1)
            
            col = row.reshape(1,3)
            self.q_ij[i,:] = np.dot(row,col).flatten()
            


        




    def initialize_global_quantities(self,parameters):

        """
        Function to set some global variables that are accessed by the methods below.

        These variables are functions of the parameters that we are trying to infer,
        or else the parameters themselves.

        They are all time-constant quantities

        """

        #GW quantities
        self.omega               = parameters["omega"]
        self.mGW,self.nGW        = principal_axes(np.pi/2.0 - parameters["dec_gw"],parameters["ra_gw"],parameters["psi_gw"])
        self.eplus,self.ecross   = polarisation_basis(self.mGW,self.nGW) 
        self.GW_direction_vector = np.cross(self.mGW,self.nGW)
        self.hp,self.hx          = h_amplitudes(parameters["Agw"],parameters["iota_gw"]) 

        #PSR quantities
        self.gamma = parameters["gamma"]
        self.n     = parameters["n"]


        #Reshape
        self.eplus_flat = self.eplus.reshape(9,1)
        self.ecross_flat = self.ecross.reshape(9,1)


        #Useful quantities
        self.dot_product = 1 + np.dot(self.GW_direction_vector,self.q.T)
        self.H_coefficient = np.real((1 - np.exp(1j*self.omega*self.pulsar_distances*self.dot_product/c)) / (2*self.dot_product))



    def F_function_new(self,x,dt):

        """
        Transition function.

        User defined function that should take the state `x` and advance it by
        `dt`.

        The state here is actually the sigma points
        """

        #Declare parameters for this function
        omega = self.omega 
        gamma = self.gamma
        n     = self.n 
        nrows = x.shape[0]
        

        #Initialize output array
        output = np.zeros_like(x) #the output should have the same shape as the input, `x`
        

      
        #Nested function - the ODE describing the evolution of the state
        def f(x,t):
            df = np.zeros(len(x))
            df[0] = omega 
            for i in range(1,len(x)):
                df[i] = -gamma*x[i]**n 
            return df

    
        for i in range(nrows): # for every sigma vector
            out_row = odeint(f,x[i,:],[0,dt])[-1]
            output[i,:] = out_row


        return output



    def F_function(self,x,dt):

        """
        Transition function.

        User defined function that should take the state `x` and advance it by
        `dt`.

        The state here is actually the sigma points
        """

        #Declare parameters for this function
        omega = self.omega 
        gamma = self.gamma
        n     = self.n 
        nrows = x.shape[0]

        

        #Initialize output array
        output = np.zeros_like(x) #the output should have the same shape as the input, `x`
        

        df = np.zeros_like(x) #the output should have the same shape as the input, `x`
        df[:,0] = np.full(nrows,omega)
        df[:,1:] = -gamma * x[:,1:]**n

        return x + dt*df

      


    def H_function(self,x):

        """
        Measurement function.

        User defined function that should take the state `x` and return the measurement

        The state here is actually the sigma points
        """

          
        #Get the hplus and hcross strains for each 2L + 1 phase 
        hplus,hcross        = self.hp*np.cos(x[:,0]),self.hx*np.sin(x[:,0]) # The time varying plus and cross GW strains


        hplus = hplus.reshape(1,len(hplus))
        hcross = hcross.reshape(1,len(hcross)) #can we avoid these reshapes?
       

        

        h_ij      = np.dot(self.eplus_flat,hplus) + np.dot(self.ecross_flat, hcross) # (9,2L+1)
        hscalar   = np.dot(h_ij.T, self.q_ij.T)
        GW_factor = 1.0 - hscalar*self.H_coefficient
        fmeasured = x[:,1:]  * GW_factor
        
    


        return fmeasured








    def H_functionold(self,x):

        """
        Measurement function.

        User defined function that should take the state `x` and return the measurement

        The state here is actually the sigma points
        """

       
        #Parameters
        omega = self.omega
        nrows = x.shape[0] #2L + 1

        
        
        #Get the hplus and hcross strains for each 2L + 1 phase 
        hplus,hcross        = self.hp*np.cos(x[:,0]),self.hx*np.sin(x[:,0]) # The time varying plus and cross GW strains

        #Get the 3x3 h_ij matrix at each 2L + 1 state. Here we have called this h_t
        h_t = np.zeros((3,3,len(hplus))) 
        for i in range(len(h_t)):
            h = h_ij(self.eplus,self.ecross,hplus[i],hcross[i])
            h_t[:,:,i] = h


        #Given the state, determine the measurement for every pulsar
        fmeasured = np.zeros((nrows,self.dims_z)) 
        for k in range(self.dims_z): #for every pulsar

            qvec = self.q[k,:]                     # Pulsar direction
            d    = self.pulsar_distances[k]        # Pulsar distance 
            fpulsar = x[:,k+1]                     # Pulsar intrinsic frequency evolution. 0th term is the GW phase 


            #First get the Einstein summation h_ij q^i q^j
            h_scalar = np.zeros(len(hplus))
            for i in range(3):
                for j in range(3):
                    value = h_t[i,j,:]*qvec[i]*qvec[j]
                    h_scalar += value

            #Now get the dot product quantity
            #Todo: we don't need to calculate this at every call! --> Define it globally
            dot_product = 1 + np.dot(self.GW_direction_vector,qvec)
            
            #This is the correction factor for the ith pulsar
            #It has length 2L + 1
            GW_factor = np.real(1.0 - 0.5*h_scalar/dot_product *(1 - np.exp(1j*omega*d*dot_product/c)))
            
            #The output
            fmeasured[:,k] = fpulsar * GW_factor

            

        return fmeasured




    def null_measurement_function(self,x):

        """
        Assume there is NO gravitational wave.
        In the Melatos forumlation, this means that the thing you measure is just the same as the hidden state, modulo some noise.
        This function just returns the measured states
        """

        return x[:,1:] 

    



    def Q_function(self,x,dt):

        """
        The user-defined process noise covariance

        Todo: how  to define this properly for non linear problems?
        Currently tuning Q by hand depending on the problem/data

        Linear approx is something like: 

            # for i in range(1,self.dims_x):
            #     Q[i,i] = sigma**2*dt*(1-self.gamma*(self.n-1)*dt*x[i]**(self.n-1))

        """
  

        Q =  np.zeros((self.dims_x,self.dims_x)) 
        for i in range(1,self.dims_x):
            Q[i,i] = 1e-5 #1e-3#0.1

      
        return Q 


    def R_function(self): 

        """
        The user-defined measurement noise covariance

        """


        return 0.0




