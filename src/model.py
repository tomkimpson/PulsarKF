import numpy as np 
from scipy.integrate import odeint

def Q_function():
    return 0

def F_function(x,parameters,dt):

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
    def f(x,t):
        df = np.zeros(len(x))
        df[0] = omega 
        for i in range(1,len(x)):
            df[i] = -gamma*x[i]**n
        return df

    return np.asarray([odeint(f,x[:,i],[0,dt]) for i in range(ncols)])











    return 0

def H_function():
    return 0 