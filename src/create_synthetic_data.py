
import sdeint
import matplotlib.pyplot as plt 
from GW import * 
from universal_constants import * 
from decimal import * 
import sys 
from configs.config import NF
import pandas as pd
import scipy
class PulsarFrequencyObservations:


    """
    Class to produce some noisy observations of the pulsar frequency,
    subject to intrinsic pulsar spin down, stochasticity and GW effects 
    """



    def __init__(self,t):

        self.t = t 
        self.dt = t[1] - t[0]

    def _frequency_ODE_f(self,x,t):
        return -self.spindown_gamma * x**self.spindown_n

    def _frequency_ODE_g(self,x,t):
        return self.spindown_sigma
      
    def create_observations(self,pulsar_parameters,GW_parameters,noise_parameters):

        #Unpack input dictionaries

        #-GW
        self.omega_GW            = GW_parameters["omega_GW"]
        self.phase_normalisation = GW_parameters["phase_normalisation"]
        self.psi_gw              = GW_parameters["psi_GW"]
        self.iota_gw             = GW_parameters["iota"] 
        self.dec_gw              = GW_parameters["dec_GW"]  
        self.ra_gw               = GW_parameters["ra_GW"]
        self.Agw                 = GW_parameters["h0"]



       


        #-Pulsars

        if pulsar_parameters["pulsar_distribution"] == "NANOGrav":

            #Load data
            df = pd.read_pickle('../data/NANOGrav_pulsars')

            self.f_psr            = df["F0"].to_numpy(dtype=NF)
            self.pulsar_distances = df["DIST"].to_numpy(dtype=NF) * 1e3 * pc #from kpc to m
            self.spindown_gamma   = df["gamma"].to_numpy(dtype=NF)
            self.spindown_n       = df["n"].to_numpy(dtype=int)
        else:
            self.f_psr            = pulsar_parameters["f_psr"]
            self.pulsar_distances = pulsar_parameters["pulsar_distances"] * 1e3 * pc #from kpc to m
            self.spindown_gamma   = pulsar_parameters["spindown_gamma"]
            self.spindown_n       = pulsar_parameters["spindown_n"]



       
        self.Npulsars = int(len(self.f_psr))
        print(f"The mean pulsar frequency over {self.Npulsars} pulsars is {np.mean(self.f_psr)}")

        
        #-Noise
        self.spindown_sigma    = np.full((self.Npulsars,1), noise_parameters["process_noise"]) # give all pulsars the same magnitude of noise
        self.measurement_noise = noise_parameters["measurement_noise"]



        #Check the types
        #print np.finfo(np.longdouble)
        function_arguments = [pulsar_parameters,GW_parameters,noise_parameters]
        for d in function_arguments:
            
            for v in d.values():
                try:
                    val_type = v.dtype
                except:
                    val_type = NF #for strings
                    pass
                
                assert val_type== NF



        print(f"The magnitude of the GW strain using these parameters is: {self.Agw}")
        print(f"The angular frequency of the GW is {self.omega_GW}")

        #Get the evolution of the intrinsic pulsar frequency by solving the Ito integral
        rng = np.random.default_rng(seed=np.random.get_state()[1][0])
        rng.normal()
        self.state_frequency = sdeint.itoint(self._frequency_ODE_f,self._frequency_ODE_g, self.f_psr, self.t,generator=rng)
        

    
        #With the evolution of the state variables, we can now get the observations
        #First define some GW related quantities
    

        m,n                 = principal_axes(NF(np.pi/2.0) - self.dec_gw,self.ra_gw,self.psi_gw)    # Get basis vectors of the GW 
        GW_direction_vector = np.cross(m,n)            
        self.hp,self.hx     = h_amplitudes(self.Agw,self.iota_gw)                                    # The GW amplitudes 
        eplus,ecross        = polarisation_basis(m,n)                                                # The polarization basis  
        
        
        Hij = self.hp * eplus + self.hx * ecross
 
        
       # hij 
        
        #hplus,hcross        = self.hp*np.cos(self.state_phase),self.hx*np.sin(self.state_phase)    # The time varying plus and cross GW strains




        
        # Get the direction vector of the pulsar
        if pulsar_parameters["pulsar_distribution"] == "random":
             dec_psr = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=self.Npulsars,dtype=NF),  
             ra_psr =  np.random.uniform(low=0.0, high=2*np.pi, size=self.Npulsars,dtype=NF),  
             self.q = pulsar_directions(NF(np.pi/2.0) - np.array(dec_psr[0]),np.array(ra_psr[0]))  
        elif pulsar_parameters["pulsar_distribution"] == "NANOGrav":
             dec_psr = df["DECJD"].to_numpy(dtype=NF)  
             ra_psr = df["RAJD"].to_numpy(dtype=NF)  
             self.q = pulsar_directions(NF(np.pi/2.0) - np.array(dec_psr),np.array(ra_psr)) 
        elif pulsar_parameters["pulsar_distribution"] == "uniform":
            self.q              = uniform_pulsar_directions(self.Npulsars)
        elif pulsar_parameters["pulsar_distribution"] == "orthogonal":
            self.q = orthogonal_pulsar_directions(self.Npulsars,GW_direction_vector)
        else:
            print(f"Pulsar distribution setting is not recognised")
            sys.exit()


        



        # h_t = np.zeros((3,3,len(self.t)),dtype=NF) #the 3x3 h_ij matrix at each moment in time 
        # for i in range(len(self.t)):
        #     hp, hx = hplus[i],hcross[i]       # get the GW amplitudes at t
        #     h = h_ij(eplus,ecross,hp,hx)      # GW strain h_ij
        #     h_t[:,:,i] = h                    # the 3x3 h_ij matrix at that moment in time 

        
        f_measured = np.zeros((len(self.t), self.Npulsars),dtype=NF) # num times x num pulsars
        
        for k in range(self.Npulsars): # For every pulsar

            qvec    = self.q[k,:]                  # Pulsar direction
            d       = self.pulsar_distances[k]     # Pulsar distance 
            fpulsar = self.state_frequency[:,k]    # Pulsar intrinsic frequency evolution 
            dot_product = 1 + np.dot(GW_direction_vector,qvec)


            h_scalar = 0
            for i in range(3):
                for j in range(3):
                    value = Hij[i,j]*qvec[i]*qvec[j]
                    h_scalar += value 
                
            
            time_variation = np.exp(-1j*self.omega_GW*self.t*dot_product + self.phase_normalisation)

            
            
            
            GW_factor = np.real(1 - NF(0.5)*h_scalar/dot_product *time_variation*(1 - np.exp(1j*self.omega_GW*d*dot_product/c)))


            
            f_measured[:,k] = fpulsar * GW_factor 
            

        #Generate some measurement noise...
        measurement_noise = np.random.normal(0, self.measurement_noise,f_measured.shape).astype(NF) # Measurement noise
        
        #...and add it to every observation
        self.observations = f_measured+measurement_noise
        self.observations_noiseless  = f_measured

      
        print("The Number format of the observations is:", self.observations.dtype)
        
    def plot_measurement_frequency(self,psr_index):


        h,w = 10,10
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(h,w),sharex=True)
        
        tplot = self.t / (60*60*24*365)

        if psr_index is None: #Plot all the pulsars, normalised
            normalized = self.observations -self.observations[0,:]
            ax.plot(tplot,normalized)
            ax.set_ylabel(r'$f_m$ [normalised]')



        else:
            ax.plot(tplot,self.observations[:,psr_index])
            print("The change in measured frequency was:", np.max(self.observations[:,psr_index])-np.min(self.observations[:,psr_index]))
            ax.set_ylabel(r'$f_m$ [Hz]')

        ax.set_xlabel('t [years]')
        ax.ticklabel_format(useOffset=False)


    def plot_state_frequency(self,psr_index):


        h,w = 10,10
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(h,w),sharex=True)
        
        tplot = self.t / (60*60*24*365)

        if psr_index is None: #Plot all the pulsars, normalised
            normalized = self.state_frequency -self.state_frequency[0,:]
            ax.plot(tplot,normalized)
            ax.set_ylabel(r'$f_p$ [normalised]')



        else:
            ax.plot(tplot,self.state_frequency[:,psr_index])
            print("The change in state frequency was:", np.max(self.state_frequency[:,psr_index])-np.min(self.state_frequency[:,psr_index]))
            ax.set_ylabel(r'$f_p$ [Hz]')
            #ax.set_ylim(np.min(self.state_frequency[:,psr_index]),np.max(self.state_frequency[:,psr_index]))

        ax.set_xlabel('t [years]')
        ax.ticklabel_format(useOffset=False)



    def plot_observations(self,psr_index,KF_predictions,savefig):


        """
        Function to plot both the observations and the state predictions of the Kalman filter
        To plot a parrticular pulsar, provide a psr_index
        """

        h,w = 10,10
        rows = 2
        cols = 1
        fig, (ax2,ax3) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)
        
        tplot = self.t / (60*60*24*365)

        #Intrinsic pulsar frequency
        if psr_index is None:
            ax2.plot(tplot,self.state_frequency,label="truth")
            if KF_predictions is not None:
                ax2.plot(tplot,KF_predictions,label="prediction") 

        else:
            ax2.plot(tplot,self.state_frequency[:,psr_index],label="truth")
            if KF_predictions is not None:
                ax2.plot(tplot,KF_predictions[:,psr_index],label="prediction") #the 0th pulsar corresponds to the 1st element of the state

            #ax2.set_ylim(np.min(KF_predictions[:,psr_index+1]),np.max(KF_predictions[:,psr_index+1]))
            #diff =   np.abs(self.state_frequency[:,psr_index] - KF_predictions[:,psr_index]) / self.state_frequency[:,psr_index]

            diff = np.abs(self.state_frequency[:,psr_index] - KF_predictions[:,psr_index])

            print("The mean error in the state prediction is:",np.sum(diff))
            
        ax2.set_ylabel(r'$f_p$ [Hz]')
        ax2.legend()

        #Measured pulsar frequency
        if psr_index is None:
            ax3.plot(tplot,self.observations,c='C2')
        else:
            ax3.plot(tplot,self.observations[:,psr_index],c='C2')



            print("Difference in the state frequency:", max(self.state_frequency[:,psr_index+1]) - min(self.state_frequency[:,psr_index+1]))
            print("Difference in the observed frequency:", max(self.observations[:,psr_index]) - min(self.observations[:,psr_index]))
            
        ax3.set_ylabel(r'$f_M$ [$\mu$Hz]')

        ax3.set_xlabel('t [years]')

 
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        if savefig is not None:
            plt.savefig(savefig,bbox_inches='tight',dpi=300)
        plt.show()
