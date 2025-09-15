import numpy as np
from src.models import (
    simulate_exponential_fixed,
    simulate_exponential_dynamic,
    simulate_logistic_fixed,
    simulate_logistic_dynamic,
    simulate_threshold_fixed,
    simulate_threshold_dynamic,
)

# --- Analysis Functions (entropy, CI, etc.) ---
def entropy(p_A):
    p_B = 1 - p_A
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -p_A * np.log(p_A) - p_B * np.log(p_B)
        h = np.nan_to_num(h)
    return h

# Run simulations across populations for a given model
def run_with_confidence_intervals(model_func, N, generations, p_A_init, reps=30):
    histories = []
    for _ in range(reps):
        history = model_func(N=N, generations=generations, p_A_init=p_A_init)
        histories.append(history)
    histories = np.array(histories)
    mean = np.mean(histories, axis=0)
    std = np.std(histories, axis=0)
    lower = mean - 1.96 * std / np.sqrt(reps)
    upper = mean + 1.96 * std / np.sqrt(reps)
    return mean, lower, upper, histories

# Plotting function
def binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Run replicates for entropy
def run_entropy_replicates(sim_func, N, reps=30, generations=5000):
    all_entropy = []
    for _ in range(reps):
        p_series = sim_func(N=N, generations=generations)
        entropy_series = [binary_entropy(p) for p in p_series]
        all_entropy.append(entropy_series)
    return np.array(all_entropy)

# Plot entropy with confidence interval
def entropy(p_A):
    p_B = 1 - p_A
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -p_A * np.log(p_A) - p_B * np.log(p_B)
        h = np.nan_to_num(h)
    return h

# Re-define the dynamic exponential mutation model with crowding
def entropy(p_A):
    p_B = 1 - p_A
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -p_A * np.log(p_A) - p_B * np.log(p_B)
        h = np.nan_to_num(h)
    return h

# ====== MODEL DEFINITIONS ======
def run_with_confidence_intervals(model_func, runs=30, N=1000, generations=5000, **kwargs):
    all_runs = []
    for _ in range(runs):
        history = model_func(N=N, generations=generations, **kwargs)
        all_runs.append(history)
    all_runs = np.array(all_runs)
    mean_p_A = np.mean(all_runs, axis=0)
    std_p_A = np.std(all_runs, axis=0)
    ci95 = 1.96 * std_p_A / np.sqrt(runs)
    return mean_p_A, ci95

# ====== PLOTTING FUNCTION (OVERLAYS) ======
def track_mutation_logistic(sim_func, N, generations, p_A_init=0.5, mu_0=0.05, beta=5, p_c=0.7, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    mutation_rates = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A

        if 'dynamic' in sim_func.__name__:
            a_i = a / (1 + alpha * i) if i > 0 else a
            d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        else:
            a_i = a
            d_i = d

        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0

        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mutation_rates.append(mu_AB)

        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5

        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else (
            "A" if np.random.rand() < mu_AB else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0

    return mutation_rates

def track_mutation_threshold(sim_func, N, generations, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.9, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    mutation_rates = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A

        if 'dynamic' in sim_func.__name__:
            a_i = a / (1 + alpha * i) if i > 0 else a
            d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        else:
            a_i = a
            d_i = d

        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0

        mu_AB = mu_0 if p_A < p_th else mu_low
        mutation_rates.append(mu_AB)

        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5

        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else (
            "A" if np.random.rand() < mu_AB else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0

    return mutation_rates

# Run logistic and threshold tracking for N=100 and N=1000
def run_model_mutation_tracking(model_name, tracker_func, sim_func_fixed, sim_func_dynamic, generations=5000):
    traces = {}
    for N in [100, 1000]:
        traces[f"{model_name} – Fixed Payoff (N={N})"] = tracker_func(
            sim_func=sim_func_fixed, N=N, generations=generations)
        traces[f"{model_name} – Dynamic Payoff (N={N})"] = tracker_func(
            sim_func=sim_func_dynamic, N=N, generations=generations)
    return traces

# logistic model
logistic_traces = run_model_mutation_tracking(
    "Logistic",
    tracker_func=track_mutation_logistic,
    sim_func_fixed=simulate_exponential_fixed,
    sim_func_dynamic=simulate_exponential_dynamic
)

# threshold model
threshold_traces = run_model_mutation_tracking(
    "Threshold",
    tracker_func=track_mutation_threshold,
    sim_func_fixed=simulate_exponential_fixed,
    sim_func_dynamic=simulate_exponential_dynamic
)

# Plot for logistic and threshold
def track_mutation_exponential(sim_func, N, generations, p_A_init=0.5, mu_0=0.05, beta=5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    mutation_rates = []
    for _ in range(generations):
        p_A = i / N
        if 'dynamic' in sim_func:
            a_i = a / (1 + alpha * i) if i > 0 else a
            d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        else:
            a_i = a
            d_i = d
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mutation_rates.append(mu_AB)
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_AB else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
    return mutation_rates

def track_mutation_logistic(sim_func, N, generations, p_A_init=0.25, mu_0=0.05, beta=5, p_c=0.7, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    mutation_rates = []
    for _ in range(generations):
        p_A = i / N
        if 'dynamic' in sim_func:
            a_i = a / (1 + alpha * i) if i > 0 else a
            d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        else:
            a_i = a
            d_i = d
        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mutation_rates.append(mu_AB)
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_AB else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
    return mutation_rates

def track_mutation_threshold(sim_func, N, generations, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.9, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    mutation_rates = []
    for _ in range(generations):
        p_A = i / N
        if 'dynamic' in sim_func:
            a_i = a / (1 + alpha * i) if i > 0 else a
            d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        else:
            a_i = a
            d_i = d
        mu_AB = mu_0 if p_A < p_th else mu_low
        mutation_rates.append(mu_AB)
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_AB else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
    return mutation_rates

# Run replicates for CI
def run_multiple_mutation_tracks(track_func, sim_func, num_reps=30, **kwargs):
    return np.array([track_func(sim_func=sim_func, **kwargs) for _ in range(num_reps)])

# Plot CI function