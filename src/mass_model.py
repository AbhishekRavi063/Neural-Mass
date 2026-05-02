from multiprocessing import connection
from numpy.typing import NDArray
import numpy as np
from population import Population
from typing import Union


class MassModel():

    def __init__(self,
                population_parameters: dict,
                connection_graph: dict,
                sampling_rate: float = 100)->None:
        """
        ARGS:
        -----
            population_parameters: A dict with the parameters of population, the unknown 
                parameters to be fitted must be filled with the choices for Optuna or None
            connection_graph: A dict of dict. Each dict determine the connection that must exist and the weights expected. None and Optuna compatible variable can be injected too

        RETURN:
            NONE
        """
        self.populations = [Population(**parameter) for parameter in population_parameters]
        self.connection_graph = connection_graph
        self.dt = 1/100
    
    def simulate(self,
                 N_points: int,
                 init_values: Union[float,list[float]])->list[float]:
        """Generate N_points through the computation of the derivative of the synaptic current

        ARGS :
        ------
            N_points: The number of point to compute
            init_value: The starting value or a list of starting values if multiple population to simulate
        
        RETURNS:
        ------
            List of float, the signal
        """
        
        # Assume each population has a voltage attribute V
        signal_matrix = np.zeros((N_points, len(self.populations)))
        # Set initial values
        for i, pop in enumerate(self.populations):
            pop.V = init_values[i]
            signal_matrix[0, i] = pop.V

        for t in range(1, N_points):
            # Compute all currents here
            # Optionally, apply connection weights from self.connection_graph
            for i, pop in enumerate(self.populations):
                dV = pop.compute_dV(self.connection_graph, self.populations)
                pop.V += self.dt * dV
                signal_matrix[t, i] = pop.V
                pop.update_states()  # Update S, etc.

        return signal_matrix

    
    def dV(self)->float:
        I_l = self.I_l()
        I_ampa = self.I_ampa()
        I_gaba = self.I_gaba()

        return - (I_l + I_ampa + I_gaba)
    
    def I_l(self)->float:
        I_l = 0
        for population in self.populations:
            I_l += population.I_l()
        return I_l

    def I_ampa(self)->float:
        I_ampa = 0
        for population in self.populations:
            I_ampa += population.I_ampa()
        return I_ampa

    def I_gaba(self)->float:
        I_gaba = 0
        for population in self.populations:
            I_gaba += population.I_gaba()
        return I_gaba
    
def Q(Q_max : float,
      V : NDArray,
      sigma : float,
      theta : float) -> NDArray:

    num = Q_max
    denum = 1 + np.exp(-(V-theta)/sigma)

    return num/denum

def alpha(gamma : float,
          t : int) -> float:
    
    A = np.arange(0, t, 1)*(gamma**2)
    B = np.exp(-gamma*np.arange(0,t,1))

    return A*B

def sm(gammas : list[float],
       N : NDArray,
       Q : NDArray) -> NDArray:
    """
    ARGS:
    -----
        gamma :
        N : A nxn matrix
        Q : A 

    RETURNS:
    -------- 
    """
    n_population = Q.shape[0]
    n_synapse_type = len(gammas)
    n_time_point = Q.shape[-1]
    out = np.zeros(shape=(n_synapse_type,
                        n_population,
                        n_time_point))
    
    for e,gamma in enumerate(gammas):
        for i in range(Q.shape[0]):
            A = alpha(gamma=gamma,t=Q.shape[-1])
            B = N[i]
            out[e,:] += np.convolve(A,B)
    
    return out
    
