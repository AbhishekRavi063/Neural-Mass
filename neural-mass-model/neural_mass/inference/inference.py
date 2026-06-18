import optuna
import numpy as np
from neural_mass.models.graph import Population, Connection, ComputationalGraph
from neural_mass.utils.metrics import calculate_rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_simulation(A, B, steps=1000, sigma=0.0, seed=None, dt=0.001):
    """Utility to run a quick simulation with given parameters."""
    cortex = Population(A=A, B=B, sigma=sigma)
    thalamus = Population(A=A, B=B, sigma=sigma)
    conns = [Connection(cortex, thalamus, weight=10.0),
             Connection(thalamus, cortex, weight=10.0)]
    graph = ComputationalGraph([cortex, thalamus], conns, dt=dt, seed=seed)

    return graph.simulate(steps=steps)[:, 0]

def find_best_parameters(
    target_signal,
    n_trials=50,
    sigma=0.0,
    seed=42,
    parameter_ranges=None,
):
    """
    Uses Optuna to find the A and B values that
    minimize the error (RMSE) compared to the target.
    """
    target_signal = np.asarray(target_signal)
    parameter_ranges = parameter_ranges or {
        "A": (1.0, 10.0),
        "B": (10.0, 60.0),
    }

    def objective(trial):
        A = trial.suggest_float("A", *parameter_ranges["A"])
        B = trial.suggest_float("B", *parameter_ranges["B"])

        sim_signal = run_simulation(
            A,
            B,
            steps=len(target_signal),
            sigma=sigma,
            seed=seed,
        )

        error = calculate_rmse(sim_signal, target_signal)
        return error

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
