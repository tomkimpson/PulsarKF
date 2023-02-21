import numpy as np 
from scipy.integrate import odeint
from universal_constants import *
import sys 

from GW import * 
from configs.config import NF

#This whole thing should really be a child class that inherits a parent





class MelatosPTAModel:


    def __init__(self,dims_x,dims_z,dictionary_of_known_quantities,f0):


        self.dims_x = dims_x
        self.dims_z = dims_z
        self.f0 = f0
        

        self.pulsar_distances = dictionary_of_known_quantities["pulsar_distances"] #* 1e3 * pc #from kpc to m
        self.measurement_noise = dictionary_of_known_quantities["measurement_noise"]**2
        self.process_noise = dictionary_of_known_quantities["process_noise"]**2
        
        
        self.q = dictionary_of_known_quantities["pulsar_directions"]
        self.N_pulsar = len(self.q)
        
        
        


        self.q_ij = np.zeros((self.N_pulsar ,9),dtype=NF) # this defines an object which for each pulsar direction (x,y,z) defines the products (xx,xy,xz,yx,yy,yz,zx,zy,zz)
        for i in range(self.N_pulsar):
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
        self.phi0                = parameters["phi0"]
        self.mGW,self.nGW        = principal_axes(NF(np.pi/2) - parameters["dec_gw"],parameters["ra_gw"],parameters["psi_gw"])
        self.GW_direction_vector = np.cross(self.mGW,self.nGW)
        self.hp,self.hx          = h_amplitudes(parameters["Agw"],parameters["iota_gw"]) 
        self.eplus,self.ecross   = polarisation_basis(self.mGW,self.nGW) 
        self.Hij                 = (self.hp * self.eplus + self.hx * self.ecross).flatten()#.reshape(9,1)
        
        
        self.hscalar             = np.zeros(self.N_pulsar)
        for i in range(len(self.hscalar)): #todo: vectorise this!
            self.hscalar[i] = np.dot(self.Hij,self.q_ij[i,:])
            
        
        
                

        #PSR quantities
        self.gamma = parameters["gamma"]
        self.n     = parameters["n"]


        #Reshape
        #self.eplus_flat = self.eplus.reshape(9,1)  #useful for vectorised operations rather than the usual 3x3 shape 
        #self.ecross_flat = self.ecross.reshape(9,1)


        #Useful quantities
        self.dot_product = 1 + np.dot(self.GW_direction_vector,self.q.T)

        self.H_coefficient = (1 - np.exp(1j*self.omega*self.pulsar_distances*self.dot_product/c)) / (2*self.dot_product)



    def F_function(self,x,dt):

        """
        Transition function.

        User defined function that should take the state `x` and advance it by
        `dt`.

        The state here is actually the sigma points
        """

        # #Declare parameters for this function
        # nrows = x.shape[0]

        
        # df = np.zeros_like(x,dtype=NF) #the output should have the same shape as the input, `x`
        # df[:,0] = np.full(nrows,self.omega)
        # df[:,1:] = -self.gamma * x[:,1:]**self.n


        df = -self.gamma*x**self.n

   
        return x + dt*df #euler step

    def H_function(self,x,t):

        """
        Measurement function.

        User defined function that should take the state `x` and return the measurement

        The state here is actually the sigma points
        """

      
       

        GW_factor = np.real(NF(1.0) - self.hscalar * np.exp(-1j*self.omega*t*(self.dot_product) + self.phi0)*self.H_coefficient)

        fmeasured = x * GW_factor
    

        return fmeasured

    def null_measurement_function(self,x,t):

        """
        Assume there is NO gravitational wave.
        In the Melatos forumlation, this means that the thing you measure is just the same as the hidden state, modulo some noise.
        This function just returns the measured states
        """

        return x
    
    def Q_function(self,x,dt):

        """
        The user-defined process noise covariance

        Todo: how  to define this properly for non linear problems?
        Currently tuning Q by hand depending on the problem/data

        Linear approx is something like: 

            # for i in range(1,self.dims_x):
            #     Q[i,i] = sigma**2*dt*(1-self.gamma*(self.n-1)*dt*x[i]**(self.n-1))

        """
  
       

       # value = self.process_noise*dt*(self.gamma*self.n*)

        coefficient = 2*self.gamma*self.n*self.f0**(self.n-1)
        expo_term = np.exp(-2*self.gamma*self.n*self.f0**(self.n-1) * dt) -1

        Q =  np.zeros((self.dims_x,self.dims_x),dtype=NF) 
        for i in range(self.dims_x):

            #Q[i,i] = -self.process_noise*dt*((self.gamma*self.n*x[i]*dt)**2 - 3.0*self.gamma*self.n*x[i]*dt + 3)
            Q[i,i] = -self.process_noise*expo_term[i]/coefficient[i]

        #for i in range(self.dims_x):

            #Q[i,i] = 1e-18 #1e-3#0.1 #1e1-14
            #Q[i,i] = 1e-13#**2 #1e-3#0.1 #1e1-14



        #print(Q)      
        #print("VALUE OF Q MATRIX")
        #print(Q)
        return Q 

    def R_function(self): 

        """
        The user-defined measurement noise covariance

        """


        return self.measurement_noise




