import numpy as np 
from scipy.integrate import odeint
from universal_constants import *
import sys 

from GW import * 
from configs.config import NF

#This whole thing should really be a child class that inherits a parent





class MelatosPTAModel:


    def __init__(self,dims_x,dims_z,dictionary_of_known_quantities):


        self.dims_x = dims_x
        self.dims_z = dims_z
        

        self.pulsar_distances = dictionary_of_known_quantities["pulsar_distances"] #* 1e3 * pc #from kpc to m
        self.measurement_noise = dictionary_of_known_quantities["measurement_noise"]**2
        
        
        
        self.q = dictionary_of_known_quantities["pulsar_directions"]
        self.N_pulsar = len(self.q)
        
        
        


        self.q_ij = np.zeros((self.N_pulsar ,9),dtype=NF) # this defines an object which for each pulsar direction (x,y,z) defines the products (xx,xy,xz,yx,yy,yz,zx,zy,zz)
        for i in range(self.N_pulsar ):
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
        self.mGW,self.nGW        = principal_axes(NF(np.pi/2) - parameters["dec_gw"],parameters["ra_gw"],parameters["psi_gw"])
        self.eplus,self.ecross   = polarisation_basis(self.mGW,self.nGW) 
        self.GW_direction_vector = np.cross(self.mGW,self.nGW)
        self.hp,self.hx          = h_amplitudes(parameters["Agw"],parameters["iota_gw"]) 

        #PSR quantities
        self.gamma = parameters["gamma"]
        self.n     = parameters["n"]


        #Reshape
        self.eplus_flat = self.eplus.reshape(9,1)  #useful for vectorised operations rather than the usual 3x3 shape 
        self.ecross_flat = self.ecross.reshape(9,1)


        #Useful quantities
        self.dot_product = 1 + np.dot(self.GW_direction_vector,self.q.T)



        self.H_coefficient = np.real((1 - np.exp(1j*self.omega*self.pulsar_distances*self.dot_product/c)) / (2*self.dot_product))



    def F_function(self,x,dt):

        """
        Transition function.

        User defined function that should take the state `x` and advance it by
        `dt`.

        The state here is actually the sigma points
        """

        #Declare parameters for this function
        nrows = x.shape[0]

        
        df = np.zeros_like(x,dtype=NF) #the output should have the same shape as the input, `x`
        df[:,0] = np.full(nrows,self.omega)
        df[:,1:] = -self.gamma * x[:,1:]**self.n

   
        return x + dt*df #euler step

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
  

        Q =  np.zeros((self.dims_x,self.dims_x),dtype=NF) 
        for i in range(1,self.dims_x):
        #for i in range(self.dims_x):

            Q[i,i] = 1e-15 #1e-3#0.1 #1e1-14

      
        return Q 

    def R_function(self): 

        """
        The user-defined measurement noise covariance

        """


        return self.measurement_noise




