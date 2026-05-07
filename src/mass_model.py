import numpy as np
from .population import Population

class MassModel:
    def __init__(self, population_configs, connections, sampling_rate=1000):
        """
        population_configs: List of dicts with parameters for each Population
        connections: List of tuples (source_idx, target_idx, weight)
        """
        self.dt = 1.0 / sampling_rate
        self.populations = [Population(**config, dt=self.dt) for config in population_configs]
        self.connections = connections
        
    def simulate(self, seconds, p_input_base=220.0):
        n_steps = int(seconds / self.dt)
        n_pops = len(self.populations)
        
        # Matrix to store results
        eeg_signals = np.zeros((n_steps, n_pops))
        
        for t in range(n_steps):
            # Calculate input for each population
            inputs = np.zeros(n_pops)
            
            # Base input + Noise for each population
            for i in range(n_pops):
                inputs[i] = p_input_base + np.random.normal(0, 10.0)
            
            # Add connections (coupling)
            for src, tgt, weight in self.connections:
                # The output of the source population (firing rate of pyramidal cells)
                # affects the input of the target population.
                out_src = self.populations[src].sigmoid(self.populations[src].y[1] - self.populations[src].y[2])
                inputs[tgt] += weight * out_src
            
            # Step each population forward
            for i in range(n_pops):
                eeg_signals[t, i] = self.populations[i].step(inputs[i])
                
        return eeg_signals
    
