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



    def initialize_global_quantities(self,parameters):

        """
        Function to set some global variables that are accessed by the methods below.

        These variables are functions of the parameters that we are trying to infer,
        or else the parameters themselves.

        They are all time-constant quantities

        """

        self.m,self.n            = principal_axes(np.pi/2.0 - parameters["dec_gw"],parameters["ra_gw"],parameters["psi_gw"])
        self.eplus,self.ecross   = polarisation_basis(self.m,self.n) 
        self.GW_direction_vector = np.cross(self.m,self.n)
        self.hp,self.hx          = h_amplitudes(parameters["Agw"],parameters["iota_gw"]) 





    def F_function(self,x,parameters,dt):

        """
        Transition function.

        User defined function that should take the state `x` and advance it by
        `dt`, subject to some `parameters`
        """

        omega = parameters["omega"]
        gamma = parameters["gamma"]
        n     = parameters["n"]

        nrows = x.shape[0]
        ncols = x.shape[1]
        output = np.zeros_like(x) #thee output should have the same shape as the input, `x`
        
        #Nested function - the ODE
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


    def H_function(self,x,parameters):

        #Load the parameters. This could also go in the __init__ since all parameters are time constant!
        omega = parameters["omega"]


        nrows = x.shape[0] #2L + 1
        ncols = x.shape[1] # L
        output = np.zeros((nrows,self.dims_z)) 

        
        #Get the hplus and hcross strains for each 2L + 1 phase 
        hplus,hcross        = self.hp*np.cos(x[:,0]),self.hx*np.sin(x[:,0]) # The time varying plus and cross GW strains

        h_t = np.zeros((3,3,len(hplus))) #the 3x3 h_ij matrix at each 2L + 1 state
        for i in range(len(h_t)):
            h = h_ij(self.eplus,self.ecross,hplus[i],hcross[i])
            h_t[:,:,i] = h


        fmeasured = np.zeros((nrows,self.dims_z))
        for k in range(self.dims_z): #for every pulsar

            qvec = self.q[k,:] # Pulsar direction
            d    = self.pulsar_distances[k]        # Pulsar distance 
            fpulsar = x[:,k+1] # Pulsar intrinsic frequency evolution. 0th term is the GW phase 


            h_scalar = np.zeros(len(hplus))
            for i in range(3):
                for j in range(3):
                    value = h_t[i,j,:]*qvec[i]*qvec[j]
                    h_scalar += value

        
            dot_product = 1 + np.dot(self.GW_direction_vector,qvec)

            GW_factor = np.real(1.0 - 0.5*h_scalar/dot_product *(1 - np.exp(1j*omega*d*dot_product/c)))
            
            fmeasured[:,k] = fpulsar * GW_factor
            




        return fmeasured



    def Q_function(self,dt):
  

        Q =  np.zeros((self.dims_x,self.dims_x)) 
        for i in range(1,self.dims_x):
            Q[i,i] = 0.001**2 *  dt

        return Q


    def R_function(self): #this could also be defined in the observations class...
        return 1e-10




