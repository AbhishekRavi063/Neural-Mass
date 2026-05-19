from neural_mass.inference import find_best_parameters, run_simulation
from neural_mass.metrics import calculate_correlation


def test_run_simulation_is_reproducible_with_seed():
    first = run_simulation(A=4.0, B=40.0, steps=200, seed=7)
    second = run_simulation(A=4.0, B=40.0, steps=200, seed=7)

    assert first.shape == (200,)
    assert (first == second).all()


def test_parameter_fitting_improves_signal_match():
    target = run_simulation(A=4.0, B=40.0, steps=400, seed=11)
    baseline = run_simulation(A=8.0, B=15.0, steps=400, seed=11)

    params, _ = find_best_parameters(target, n_trials=25, seed=11)
    fitted = run_simulation(params["A"], params["B"], steps=400, seed=11)

    assert calculate_correlation(fitted, target) > calculate_correlation(baseline, target)
