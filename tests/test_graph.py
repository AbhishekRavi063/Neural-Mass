import numpy as np
import pytest

from neural_mass.graph import ComputationalGraph, Connection, Population


def test_population_output_and_reset():
    population = Population(name="Cortex")
    population.y = np.array([0.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    assert population.output == 2.0

    population.reset()

    assert np.allclose(population.y, np.zeros(6))


def test_graph_simulation_shape_and_reproducibility():
    def build_graph():
        cortex = Population(name="Cortex")
        thalamus = Population(name="Thalamus")
        connections = [
            Connection(cortex, thalamus, weight=10.0),
            Connection(thalamus, cortex, weight=10.0),
        ]
        return ComputationalGraph([cortex, thalamus], connections, dt=0.001, seed=123)

    first = build_graph().simulate(steps=100)
    second = build_graph().simulate(steps=100)

    assert first.shape == (100, 2)
    assert np.allclose(first, second)
    assert np.isfinite(first).all()


def test_graph_dict_output_uses_population_names():
    cortex = Population(name="Cortex")
    graph = ComputationalGraph([cortex], [], seed=42)

    result = graph.simulate(steps=10, as_dict=True)

    assert list(result) == ["Cortex"]
    assert result["Cortex"].shape == (10,)


def test_simulate_requires_positive_duration():
    graph = ComputationalGraph([Population()], [])

    with pytest.raises(ValueError):
        graph.simulate()

    with pytest.raises(ValueError):
        graph.simulate(steps=0)
