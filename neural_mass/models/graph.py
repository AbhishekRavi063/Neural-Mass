from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class Population:
    """
    Represents one Jansen-Rit neural mass population.

    State variables (y):
        y[0..2]: post-synaptic potentials (pyramidal, excitatory, inhibitory)
        y[3..5]: first derivatives of those potentials
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
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.C = C
        self.v0 = v0
        self.e0 = e0
        self.r = r
        # Connectivity constants (C1-C4) derived from C (Jansen & Rit 1995)
        self.C1 = C
        self.C2 = 0.8 * C
        self.C3 = 0.25 * C
        self.C4 = 0.25 * C
        self.y = np.zeros(6)

    def sigmoid(self, v):
        return (2 * self.e0) / (1 + np.exp(self.r * (self.v0 - v)))

    def compute_dv(
        self, p_input: float, state: NDArray | None = None
    ) -> NDArray[np.float64]:
        """
        Compute the Jansen-Rit six-state derivative.

        p_input is the total input (network coupling + external drive).
        state optionally overrides self.y for RK4 sub-steps.
        """
        y = state if state is not None else self.y
        dy = np.zeros(6)
        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]
        dy[3] = (
            self.A * self.a * self.sigmoid(y[1] - y[2])
            - 2 * self.a * y[3]
            - (self.a ** 2) * y[0]
        )
        dy[4] = (
            self.A * self.a * (p_input + self.C2 * self.sigmoid(self.C1 * y[0]))
            - 2 * self.a * y[4]
            - (self.a ** 2) * y[1]
        )
        dy[5] = (
            self.B * self.b * (self.C4 * self.sigmoid(self.C3 * y[0]))
            - 2 * self.b * y[5]
            - (self.b ** 2) * y[2]
        )
        return dy

    def compute_dV(self, p_input):
        """Backward-compatible alias."""
        return self.compute_dv(p_input)

    def update_state(self, dy, dt):
        self.y += dy * dt

    def reset(self):
        self.y = np.zeros(6)

    def parameters(self):
        return {
            "N": self.N, "tau": self.tau, "A": self.A, "B": self.B,
            "a": self.a, "b": self.b, "C": self.C, "v0": self.v0,
            "e0": self.e0, "r": self.r, "sigma": self.sigma,
        }

    @property
    def output(self):
        return self.y[1] - self.y[2]


@dataclass
class Connection:
    """Directed coupling from one population to another."""

    source: Population
    target: Population
    weight: float = 1.0


class ComputationalGraph:
    """
    Orchestrates simulation of a network of neural mass populations
    using 4th-order Runge-Kutta integration for the deterministic dynamics.
    Stochastic drive is applied once per step (Euler-Maruyama).
    """

    def __init__(
        self,
        populations,
        connections,
        dt=0.001,
        input_mean=220.0,
        input_std=22.0,
        seed=None,
    ):
        self.populations = list(populations)
        self.connections = list(connections)
        self.dt = dt
        self.input_mean = input_mean
        self.input_std = input_std
        self.rng = np.random.default_rng(seed)

    def _compute_derivs(self, states: dict, drives: dict) -> dict:
        """Compute derivatives for all populations given states and pre-sampled drives."""
        network_inputs: dict = {p: 0.0 for p in self.populations}
        for conn in self.connections:
            src_output = float(states[conn.source][1] - states[conn.source][2])
            network_inputs[conn.target] += conn.weight * src_output
        return {
            p: p.compute_dv(network_inputs[p] + drives[p], states[p])
            for p in self.populations
        }

    def step(self):
        # Sample stochastic drives once per step to keep Euler-Maruyama consistency.
        drives = {
            p: (
                self.rng.normal(self.input_mean, self.input_std)
                + self.rng.normal(0.0, p.sigma)
            )
            for p in self.populations
        }
        s0 = {p: p.y.copy() for p in self.populations}
        dt = self.dt

        k1 = self._compute_derivs(s0, drives)
        k2 = self._compute_derivs(
            {p: s0[p] + 0.5 * dt * k1[p] for p in self.populations}, drives
        )
        k3 = self._compute_derivs(
            {p: s0[p] + 0.5 * dt * k2[p] for p in self.populations}, drives
        )
        k4 = self._compute_derivs(
            {p: s0[p] + dt * k3[p] for p in self.populations}, drives
        )

        for p in self.populations:
            p.y = s0[p] + (dt / 6.0) * (k1[p] + 2.0 * k2[p] + 2.0 * k3[p] + k4[p])

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
