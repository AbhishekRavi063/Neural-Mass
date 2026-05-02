# Neural Mass Models

Neural mass models are simplified descriptions of large populations of neurons, used to relate mesoscopic dynamics to signals such as EEG. This project starts from two building blocks—`Population` and `Connection`—and a `ComputationalGraph` that wires them into a network and advances them in time. The abstractions can be extended later (e.g. richer dynamics, delays, stochasticity).

## Population class

The `Population` class represents a population of neurons. It is intended to hold **local** parameters and **state** for that mass (e.g. mean activity or firing rate), while **inter-population** coupling lives on `Connection` objects.

Attributes:

- `N`: number of neurons in the population
- `tau`: time constant of the mass
- `G`: connection weights (e.g. lumped or recurrent weights associated with this population, depending on the chosen equations)
- `E`: excitatory connection weights (population-level excitatory contribution, if your parametrization separates it)
- `gamma`: inhibitory connection weights (population-level inhibitory contribution, if your parametrization separates it)

How you map `G`, `E`, and `gamma` to a specific mass model (Wilson–Cowan, Jansen–Rit, etc.) is left to the implementation; the split above keeps room for both self-connections and typed E/I gains.

## Connection class

The `Connection` class represents a **directed** link from one population to another.

Attributes:

- `source`: source population
- `target`: target population
- `weight`: scalar (or, in a generalized version, matrix) coupling strength from source output to target input

## ComputationalGraph class

The `ComputationalGraph` binds populations and connections into a **directed graph** and runs a discrete-time simulation: each step aggregates inputs along edges, then updates each population.

Attributes:

- `populations`: list (or mapping) of `Population` instances that belong to this network
- `connections`: list of `Connection` instances; each `source` and `target` should refer to populations in `populations`
- `dt`: integration time step for explicit stepping
- `Ne`: number of excitatory neurons (network-level count; useful when the graph implements a standard E/I mass model)
- `Se`: excitatory activity (network-level excitatory state or aggregate you choose to track alongside per-population state)
- `Ni`: number of inhibitory neurons (network-level count)
- `Si`: inhibitory activity (network-level inhibitory state or aggregate)

The graph is responsible for **orchestration**, not for replacing the differential equations inside each `Population`. A typical step is:

1. Initialize an input accumulator per population (e.g. zero).
2. For each `Connection`, add `weight * source_output` to the accumulator of `target`.
3. For each `Population`, advance its state by `dt` using its local parameters and the accumulated input.

If the network has algebraic loops within one time step, you may need an implicit scheme or fixed-point iteration; that policy belongs in the graph’s step method.

## Usage

Illustrative pattern (define these classes in a module such as `src/graph.py` and import accordingly):

```python
from src.graph import Population, Connection, ComputationalGraph

# Define populations (parameters depend on your mass model)
e_pop = Population(N=8000, tau=0.01, G=1.0, E=1.0, gamma=0.0)
i_pop = Population(N=2000, tau=0.02, G=1.0, E=0.0, gamma=1.0)

# Directed edges: source -> target
conns = [
    Connection(source=e_pop, target=e_pop, weight=0.5),
    Connection(source=e_pop, target=i_pop, weight=0.3),
    Connection(source=i_pop, target=e_pop, weight=-0.4),
    Connection(source=i_pop, target=i_pop, weight=-0.2),
]

graph = ComputationalGraph(
    populations=[e_pop, i_pop],
    connections=conns,
    dt=1e-3,
    Ne=e_pop.N,
    Se=0.0,
    Ni=i_pop.N,
    Si=0.0,
)

for _ in range(1000):
    graph.step()  # implements aggregate inputs + per-population integration
```

Adjust `Population` fields, `step()` internals, and how `Se` / `Si` are updated to match the equations you implement.
