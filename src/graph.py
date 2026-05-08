import numpy as np

class Population:
    """
    Represents a population of neurons (Jansen-Rit unit).
    Holds local parameters and state.
    """
    def __init__(self, N=1000, tau=0.01, A=3.25, B=22.0, a=100.0, b=50.0, C=135.0, v0=6.0, e0=2.5, r=0.56):
        # Specs from README
        self.N = N
        self.tau = tau
        
        # Jansen-Rit specific parameters
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.C = C
        self.v0 = v0
        self.e0 = e0
        self.r = r
        
        # Connectivity constants (C1-C4) derived from C
        self.C1 = C
        self.C2 = 0.8 * C
        self.C3 = 0.25 * C
        self.C4 = 0.25 * C
        
        # State variables [y0, y1, y2, y3, y4, y5]
        self.y = np.zeros(6)

    def sigmoid(self, v):
        return (2 * self.e0) / (1 + np.exp(self.r * (self.v0 - v)))

    def compute_dV(self, p_input):
        """
        The Jansen-Rit 6-state ODEs.
        p_input is the input from other populations + noise.
        """
        y = self.y
        dy = np.zeros(6)
        
        # y0, y1, y2 are potentials; y3, y4, y5 are their derivatives
        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]
        
        # Equations based on Jansen-Rit model
        dy[3] = self.A * self.a * self.sigmoid(y[1] - y[2]) - 2 * self.a * y[3] - (self.a**2) * y[0]
        dy[4] = self.A * self.a * (p_input + self.C2 * self.sigmoid(self.C1 * y[0])) - 2 * self.a * y[4] - (self.a**2) * y[1]
        dy[5] = self.B * self.b * (self.C4 * self.sigmoid(self.C3 * y[0])) - 2 * self.b * y[5] - (self.b**2) * y[2]
        
        return dy

    def update_state(self, dy, dt):
        self.y += dy * dt

    @property
    def output(self):
        """The output of this population (pyramidal potential)"""
        return self.y[1] - self.y[2]

class Connection:
    """
    Represents a directed link from one population to another.
    """
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight

class ComputationalGraph:
    """
    Orchestrates the simulation of a network of populations.
    """
    def __init__(self, populations, connections, dt=0.01, Ne=0, Se=0.0, Ni=0, Si=0.0):
        self.populations = populations
        self.connections = connections
        self.dt = dt
        
        # Network-level trackers (Specs from README)
        self.Ne = Ne  # Excitatory neuron count
        self.Se = Se  # Excitatory state/activity
        self.Ni = Ni  # Inhibitory neuron count
        self.Si = Si  # Inhibitory state/activity
        
    def step(self):
        # 1. Initialize input accumulators (using a dictionary to map population to input)
        inputs = {p: 0.0 for p in self.populations}
        
        # 2. Accumulate inputs along connections
        for conn in self.connections:
            inputs[conn.target] += conn.weight * conn.source.output
            
        # 3. Add background noise (p_input typically includes a constant + noise)
        # For simplicity, we'll assume a mean input of 220 with standard deviation 22
        for p in self.populations:
            p_input = inputs[p] + np.random.normal(220, 22)
            dy = p.compute_dV(p_input)
            p.update_state(dy, self.dt)

    def simulate(self, seconds):
        steps = int(seconds / self.dt)
        history = []
        for _ in range(steps):
            self.step()
            # Record the output (EEG-like signal) of the first population
            history.append([p.output for p in self.populations])
        return np.array(history)
