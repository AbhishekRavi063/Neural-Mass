import numpy as np

class Population:
    """
    Represents a single Jansen-Rit neural mass unit.
    This unit contains three populations: Pyramidal, Excitatory Interneurons, and Inhibitory Interneurons.
    """
    def __init__(self, 
                 A=3.25, B=22.0, a=100.0, b=50.0, 
                 C=135.0, v0=6.0, e0=2.5, r=0.56,
                 dt=0.01):
        
        # Parameters
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.C = C
        self.v0 = v0
        self.e0 = e0
        self.r = r
        self.dt = dt
        
        # Connectivity constants (from Paper 1)
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
        Calculates the derivatives for the 6 state variables.
        """
        dy = np.zeros(6)
        
        dy[0] = self.y[3]
        dy[1] = self.y[4]
        dy[2] = self.y[5]
        
        dy[3] = self.A * self.a * self.sigmoid(self.y[1] - self.y[2]) - 2 * self.a * self.y[3] - self.a**2 * self.y[0]
        dy[4] = self.A * self.a * (p_input + self.C2 * self.sigmoid(self.C1 * self.y[0])) - 2 * self.a * self.y[4] - self.a**2 * self.y[1]
        dy[5] = self.B * self.b * (self.C4 * self.sigmoid(self.C3 * self.y[0])) - 2 * self.b * self.y[5] - self.b**2 * self.y[2]
        
        return dy

    def step(self, p_input):
        """
        Advances the state of the population by dt using Euler integration.
        """
        dy = self.compute_dV(p_input)
        self.y += dy * self.dt
        return self.y[1] - self.y[2] # Return the EEG-like signal (Potential)