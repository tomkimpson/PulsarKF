from numpy import sin,cos
from universal_constants import *
import numpy as np 

def principal_axes(theta,phi,psi):

    """
    From the GW source location and the polarisation angle, determine the principal axes
    """

    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = np.array([m1,m2,m3])

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = np.array([n1,n2,n3])

  
    return m,n


def polarisation_basis(m,n):


    e_plus  = np.zeros((3,3))
    e_cross = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_plus[i,j]  = m[i]*m[j] -n[i]*n[j]
            e_cross[i,j] = m[i]*n[j] -n[i]*m[j]

    return e_plus, e_cross


def h_amplitudes(Agw,iota_gw):
        hplus  = Agw*(1+np.cos(iota_gw)**2)
        hcross = Agw*(-2*np.cos(iota_gw))

        return hplus,hcross

def h_ij(e_plus,e_cross,hplus,hcross):
    h = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            h[i,j]  = e_plus[i,j]*hplus + e_cross[i,j]*hcross
    return h



def pulsar_directions(polar_angle,azimuth_angle):

    """
    Given a latitude (measured from the zenith) and an azimuth define a unit vector
    """

    q = np.zeros((len(polar_angle),3))

    for i in range(len(polar_angle)):
        q[i,0] = np.sin(polar_angle[i])*np.cos(azimuth_angle[i])
        q[i,1] = np.sin(polar_angle[i])*np.sin(azimuth_angle[i])
        q[i,2] = np.cos(polar_angle[i])
    
    return q 












