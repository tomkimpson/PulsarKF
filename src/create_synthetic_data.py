
import sdeint
import matplotlib.pyplot as plt 
from GW import * 
from universal_constants import * 

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

        #-Pulsars
        self.f_psr   = pulsar_parameters["f_psr"]
        self.dec_psr = pulsar_parameters["dec_psr"]
        self.ra_psr  = pulsar_parameters["ra_psr"]
        self.pulsar_distances = pulsar_parameters["pulsar_distances"] * 1e3 * pc #from kpc to m
        self.spindown_gamma = pulsar_parameters["spindown_gamma"]
        self.spindown_n = pulsar_parameters["spindown_n"]

        self.Npulsars = len(self.f_psr)


        #-GW
        self.omega_GW            = GW_parameters["omega_GW"]
        self.phase_normalisation = GW_parameters["phase_normalisation"]
        self.psi_gw              = GW_parameters["psi_GW"]
        self.iota_gw             = GW_parameters["iota"]  
        self.dec_gw              = GW_parameters["dec_GW"]  
        self.ra_gw                = GW_parameters["ra_GW"] 
        
        #-Noise
        self.spindown_sigma    = noise_parameters["process_noise"].reshape(self.Npulsars,1)
        self.measurement_noise = noise_parameters["measurement_noise"]

        #Define some extra useful quantities
        f_gw            = self.omega_GW/(2*np.pi)   
        m1              = GW_parameters["m1"]*Msolar            # mass of object 1 in kg
        m2              = GW_parameters["m2"]*Msolar            # mass of object 2 in kg
        chirp_mass      =  m1**(3/5) * m2**(3/5)/(m1+m2)**(1/5) #chirp mass in kg
        Dl              =  GW_parameters["Dl"]*1e9 * pc         #distance, converted from Gpc to m
        Agw             = 2 * chirp_mass**(5/3)/Dl * (np.pi*f_gw)**(2/3) #amplitude parameter
        convert_to_SI   = G**(5/3) * c**(-4.0)
        Agw             = Agw*convert_to_SI # this is now a dimensionless quantity


        #Get the evolution of the intrinsic pulsar frequency by solving the Ito integral
        self.state_frequency = sdeint.itoint(self._frequency_ODE_f,self._frequency_ODE_g, self.f_psr, self.t)
 
        #The GW phase timeseries is trivial for a monochromatic source
        self.state_phase = self.omega_GW*self.t + self.phase_normalisation


        #With the evolution of the state variables, we can now get the observations
        #First define some GW related quantities, with the ultimate aim of getting the 3x3 matrix h_ij for every timestep
        
        self.q              = pulsar_directions(np.pi/2.0 - self.dec_psr,self.ra_psr)           # Get the direction vector of the pulsar
        m,n                 = principal_axes(np.pi/2.0 - self.dec_gw,self.ra_gw,self.psi_gw)    # Get basis vectors of the GW 
        GW_direction_vector = np.cross(m,n)                                                     # The GW propagation direction
        self.hp,self.hx     = h_amplitudes(Agw,self.iota_gw)                                    # The GW amplitudes 
        eplus,ecross        = polarisation_basis(m,n)                                           # The polarization basis  
        hplus,hcross        = self.hp*np.cos(self.state_phase),self.hx*np.sin(self.state_phase) # The time varying plus and cross GW strains

        h_t = np.zeros((3,3,len(self.t))) #the 3x3 h_ij matrix at each moment in time 
        for i in range(len(self.t)):
            hp, hx = hplus[i],hcross[i]       # get the GW amplitudes at t
            h = h_ij(eplus,ecross,hp,hx)      # GW strain h_ij
            h_t[:,:,i] = h                    # the 3x3 h_ij matrix at that moment in time 

        #Cool! We now have h_t and we can determine the frequency timeseries as measured at Earth as follows
        f_measured = np.zeros((len(self.t), self.Npulsars)) # num times x num pulsars
        for k in range(self.Npulsars): # For every pulsar

            qvec = self.q[k,:]                  # Pulsar direction
            d = self.pulsar_distances[k]        # Pulsar distance 
            fpulsar = self.state_frequency[:,k] # Pulsar intrinsic frequency evolution 

            h_scalar = np.zeros(len(self.t)) #summation quantity. I'm sure there is a Pythonic way to do this
            for i in range(3):
                for j in range(3):
                    value = h_t[i,j,:]*qvec[i]*qvec[j]
                    h_scalar += value 
                
            
            dot_product = 1 + np.dot(GW_direction_vector,qvec)
            GW_factor = np.real(1.0 - 0.5*h_scalar/dot_product *(1 - np.exp(1j*self.omega_GW*d*dot_product/c)))
            f_measured[:,k] = fpulsar * GW_factor 

        #Generate some measurement noise...
        measurement_noise = np.random.normal(0, self.measurement_noise,f_measured.shape) # Measurement noise
        
        #...and add it to every observation
        self.observations = f_measured#+measurement_noise
        self.observations_noiseless  = f_measured


    def plot_observations(self,psr_index):

        h,w = 10,10
        rows = 4
        cols = 1
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)
        
        tplot = self.t / (60*60*24*365)
        #GW phase
        ax1.plot(tplot,self.state_phase)
        ax1.set_ylabel(r'$\Phi$')

        #Intrinsic pulsar frequency
        if psr_index is None:
            ax2.plot(tplot,self.state_frequency)
        else:
            ax2.plot(tplot,self.state_frequency[:,psr_index])

        ax2.set_ylabel(r'$f_p$ [Hz]')

        #Measured pulsar frequency
        if psr_index is None:
            ax3.plot(tplot,self.observations)
        else:
            ax3.plot(tplot,self.observations[:,psr_index])
        ax3.set_ylabel(r'$f_M$ [Hz]')

        #Difference
        if psr_index is None:
            ax4.plot(tplot,self.observations-self.state_frequency)
        else:
            ax3.plot(tplot,self.observations[:,psr_index] - self.state_frequency[:,psr_index])
        
        ax4.set_ylabel(r'$\Delta$')
        ax4.set_xlabel('t [years]')

 
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()





