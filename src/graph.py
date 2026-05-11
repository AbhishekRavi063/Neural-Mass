from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

class Population:
    """
    Represents one Jansen-Rit neural mass population.

    The six state variables are:
    y0, y1, y2: post-synaptic potentials
    y3, y4, y5: first derivatives of those potentials
    """
    def __init__(
        self,
        N=1000,
        tau=0.01,
        A=3.25,
        B=22.0,
        a=100.0,
        b=50.0,
        C=135.0,
        v0=6.0,
        e0=2.5,
        r=0.56,
        name="Population",
        sigma=0.0,
    ):
        self.N = N
        self.tau = tau
        self.name = name
        self.sigma = sigma
        
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
        
        self.y = np.zeros(6)

    def sigmoid(self, v):
        return (2 * self.e0) / (1 + np.exp(self.r * (self.v0 - v)))

    def compute_dv(self, p_input: float) -> NDArray[np.float64]:
        """
        Compute the Jansen-Rit six-state derivative for one time step.

        p_input is the external input arriving at this population after network
        coupling and stochastic drive have already been applied.
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

    def compute_dV(self, p_input):
        """Backward-compatible alias for earlier scripts."""
        return self.compute_dv(p_input)

    def update_state(self, dy, dt):
        self.y += dy * dt

    def reset(self):
        self.y = np.zeros(6)

    def parameters(self):
        return {
            "N": self.N,
            "tau": self.tau,
            "A": self.A,
            "B": self.B,
            "a": self.a,
            "b": self.b,
            "C": self.C,
            "v0": self.v0,
            "e0": self.e0,
            "r": self.r,
            "sigma": self.sigma,
        }

    @property
    def output(self):
        """The output of this population (pyramidal potential)"""
        return self.y[1] - self.y[2]

@dataclass
class Connection:
    """
    Directed coupling from one population to another.
    """
    source: Population
    target: Population
    weight: float = 1.0

class ComputationalGraph:
    """
    Orchestrates simulation of a network of neural mass populations.
    """
    def __init__(
        self,
        populations,
        connections,
        dt=0.001,
        input_mean=220.0,
        input_std=22.0,
        seed=None,
        Ne=0,
        Se=0.0,
        Ni=0,
        Si=0.0,
    ):
        self.populations = list(populations)
        self.connections = list(connections)
        self.dt = dt
        self.input_mean = input_mean
        self.input_std = input_std
        self.rng = np.random.default_rng(seed)
        
        self.Ne = Ne  # Excitatory neuron count
        self.Se = Se  # Excitatory state/activity
        self.Ni = Ni  # Inhibitory neuron count
        self.Si = Si  # Inhibitory state/activity
        
    def step(self):
        inputs = {p: 0.0 for p in self.populations}
        
        for conn in self.connections:
            inputs[conn.target] += conn.weight * conn.source.output
            
        for p in self.populations:
            external_drive = self.rng.normal(self.input_mean, self.input_std)
            population_noise = self.rng.normal(0.0, p.sigma)
            p_input = inputs[p] + external_drive + population_noise
            dy = p.compute_dv(p_input)
            p.update_state(dy, self.dt)

    def reset(self):
        for p in self.populations:
            p.reset()

    def simulate(self, seconds=None, steps=None, as_dict=False):
        if steps is None:
            if seconds is None:
                raise ValueError("Provide either seconds or steps.")
            steps = int(seconds / self.dt)
        if steps <= 0:
            raise ValueError("Simulation length must be positive.")

        history = []
        for _ in range(steps):
            self.step()
            history.append([p.output for p in self.populations])
        signal = np.array(history)
        if as_dict:
            return {p.name: signal[:, idx] for idx, p in enumerate(self.populations)}
        return signal
