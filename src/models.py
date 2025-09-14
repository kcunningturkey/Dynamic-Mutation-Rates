# --- Simulation Models ---
import numpy as np
def simulate_exponential_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""###**Exponential Mutation, Dynamic Payoff**"""

def simulate_exponential_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""---
##**Logistic Model**

\begin{equation}
    \mu_{AB}(t) = \frac{\mu_0}{1 + e^{\beta(p_A-p_c)}}
\end{equation}
<br>
\begin{equation}
    \mu_{BA}(t) = \frac{\mu_0}{1 + e^{\beta(p_B-p_c)}}
\end{equation}
where:

   * $p_c$ is the critical proportion where mutation effects start significantly decreasing.
   * $\beta$ controls the steepness of decay.

###**Logistic Mutation, Fixed Payoff**
"""

def simulate_logistic_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, p_c=0.5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mu_BA = mu_0 / (1 + np.exp(beta * (p_B - p_c)))
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""###**Logistic Mutation, Dynamic Payoff**"""

def simulate_logistic_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, p_c=0.5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mu_BA = mu_0 / (1 + np.exp(beta * (p_B - p_c)))
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""---
##**Threshold Model**

\begin{equation}
    \mu_{AB}(t) = \left\{
\begin{array}{ll}
\mu_0, & \text{if } p_A < p_{threshold} \\
\mu_{low},  & \text{if } p_A \geq p_{threshold}
\end{array}
\right.
\end{equation}
<br>
\begin{equation}
    \mu_{BA}(t) = \left\{
\begin{array}{ll}
\mu_0, & \text{if } p_B < p_{threshold} \\
\mu_{low},  & \text{if } p_B \geq p_{threshold}
\end{array}
\right.
\end{equation}
where:

   * $\mu_{low} \ll \mu_0$
   * $p_{threshold}$ is the critical strategy proportion where mutation rate shifts.

###**Threshold Mutation, Fixed Payoff**
"""

def simulate_threshold_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 if p_A < p_th else mu_low
        mu_BA = mu_0 if p_B < p_th else mu_low
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""###**Threshold Mutation, Dynamic Payoff**"""

def simulate_threshold_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []

    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 if p_A < p_th else mu_low
        mu_BA = mu_0 if p_B < p_th else mu_low
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

"""##N = 10, 100, 1000, 10,000 Models"""

# Now that all model functions are defined, rerun the analysis across population sizes
# and collect summary statistics

# Re-import necessary tools for plotting and display
import pandas as pd

# Define entropy calculation
def wrapped_simulate_exponential_fixed(**kwargs):
    return simulate_exponential_fixed(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        beta=kwargs.get("beta", 5)
    )

def wrapped_simulate_exponential_dynamic(**kwargs):
    return simulate_exponential_dynamic(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        beta=kwargs.get("beta", 5),
        alpha=kwargs.get("alpha", 0.05)
    )

def wrapped_simulate_logistic_fixed(**kwargs):
    return simulate_logistic_fixed(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        beta=kwargs.get("beta", 5),
        p_c=kwargs.get("p_c", 0.5)
    )

def wrapped_simulate_logistic_dynamic(**kwargs):
    return simulate_logistic_dynamic(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        beta=kwargs.get("beta", 5),
        p_c=kwargs.get("p_c", 0.5),
        alpha=kwargs.get("alpha", 0.05)
    )

def wrapped_simulate_threshold_fixed(**kwargs):
    return simulate_threshold_fixed(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        mu_low=kwargs.get("mu_low", 0.001),
        p_th=kwargs.get("p_th", 0.5)
    )

def wrapped_simulate_threshold_dynamic(**kwargs):
    return simulate_threshold_dynamic(
        N=kwargs.get("N", 100),
        generations=kwargs.get("generations", 5000),
        p_A_init=kwargs.get("p_A_init", 0.5),
        mu_0=kwargs.get("mu_0", 0.05),
        mu_low=kwargs.get("mu_low", 0.001),
        p_th=kwargs.get("p_th", 0.5),
        alpha=kwargs.get("alpha", 0.05)
    )

# Replace model dict with wrapped versions
models_to_test = {
    "Exponential (Fixed)": wrapped_simulate_exponential_fixed,
    "Exponential (Dynamic)": wrapped_simulate_exponential_dynamic,
    "Logistic (Fixed)": wrapped_simulate_logistic_fixed,
    "Logistic (Dynamic)": wrapped_simulate_logistic_dynamic,
    "Threshold (Fixed)": wrapped_simulate_threshold_fixed,
    "Threshold (Dynamic)": wrapped_simulate_threshold_dynamic
}

# @title
# Parameter values
mu_0_values = [0.01, 0.05, 0.10]
p_A_init_values = [0.1, 0.3, 0.5, 0.7, 0.9]
beta_values = [1, 5, 10]
alpha_values = [0, 0.001, 0.01, 0.10]

# Fixed parameters
shared_params = {
    "generations": 5000,
    "p_c": 0.5,
    "p_th": 0.5,
    "mu_low": 0.001
}

# Fixed population size
fixed_N = 1000

# Storage for results
simulated_variants_fixed_N = {
    "vary_mu_0": [],
    "vary_p_A_init": [],
    "vary_beta": [],
    "vary_alpha": []
}

# Wrapped models
models_to_test = {
    "Exponential (Fixed)": wrapped_simulate_exponential_fixed,
    "Exponential (Dynamic)": wrapped_simulate_exponential_dynamic,
    "Logistic (Fixed)": wrapped_simulate_logistic_fixed,
    "Logistic (Dynamic)": wrapped_simulate_logistic_dynamic,
    "Threshold (Fixed)": wrapped_simulate_threshold_fixed,
    "Threshold (Dynamic)": wrapped_simulate_threshold_dynamic
}

# Vary mu_0
for model_name, model_func in models_to_test.items():
    for mu_0 in mu_0_values:
        result = model_func(N=fixed_N, mu_0=mu_0, **shared_params)
        simulated_variants_fixed_N["vary_mu_0"].append({
            "model": model_name, "mu_0": mu_0, "history": result
        })

# Vary p_A_init
for model_name, model_func in models_to_test.items():
    for p_A_init in p_A_init_values:
        result = model_func(N=fixed_N, p_A_init=p_A_init, **shared_params)
        simulated_variants_fixed_N["vary_p_A_init"].append({
            "model": model_name, "p_A_init": p_A_init, "history": result
        })

# Vary beta
for model_name, model_func in models_to_test.items():
    for beta in beta_values:
        result = model_func(N=fixed_N, beta=beta, **shared_params)
        simulated_variants_fixed_N["vary_beta"].append({
            "model": model_name, "beta": beta, "history": result
        })

# Vary alpha
for model_name, model_func in models_to_test.items():
    for alpha in alpha_values:
        result = model_func(N=fixed_N, alpha=alpha, **shared_params)
        simulated_variants_fixed_N["vary_alpha"].append({
            "model": model_name, "alpha": alpha, "history": result
        })

# Output summary
variant_summary_fixed_N = {
    k: f"{len(v)} simulations at N=1000" for k, v in simulated_variants_fixed_N.items()
}
print(variant_summary_fixed_N)

# @title
import matplotlib.pyplot as plt

# Utility function to extract label and plot a group of simulation histories
def wrapped_simulate_exponential_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, alpha=0.05, **kwargs):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

# Function to compute mean and CI from multiple runs
def wrapped_simulate_exponential_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, **kwargs):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

def wrapped_simulate_exponential_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, alpha=0.05, **kwargs):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

# ====== CONFIDENCE INTERVAL FUNCTION ======
def simulate_exponential_fixed(): pass
def simulate_exponential_dynamic(): pass

def simulate_exponential_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

def simulate_logistic_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, p_c=0.5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mu_BA = mu_0 / (1 + np.exp(beta * (p_B - p_c)))
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

def simulate_threshold_fixed(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.5):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        f_A = (a * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 if p_A < p_th else mu_low
        mu_BA = mu_0 if p_B < p_th else mu_low
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

# Runner for multiple simulations
def simulate_exponential_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 * np.exp(-beta * p_A)
        mu_BA = mu_0 * np.exp(-beta * p_B)
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

def simulate_logistic_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, beta=5, p_c=0.5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 / (1 + np.exp(beta * (p_A - p_c)))
        mu_BA = mu_0 / (1 + np.exp(beta * (p_B - p_c)))
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

def simulate_threshold_dynamic(N=100, generations=5000, p_A_init=0.5, mu_0=0.05, mu_low=0.001, p_th=0.5, alpha=0.05):
    i = int(p_A_init * N)
    a, b, d = 3.0, 2.0, 1.0
    history = []
    for _ in range(generations):
        p_A = i / N
        p_B = 1 - p_A
        a_i = a / (1 + alpha * i) if i > 0 else a
        d_i = d / (1 + alpha * (N - i)) if (N - i) > 0 else d
        f_A = (a_i * (i - 1) + b * (N - i)) / i if i > 0 else 0
        f_B = (b * i + d_i * (N - i - 1)) / (N - i) if i < N else 0
        mu_AB = mu_0 if p_A < p_th else mu_low
        mu_BA = mu_0 if p_B < p_th else mu_low
        total_fit = i * f_A + (N - i) * f_B
        prob_A = (i * f_A) / total_fit if total_fit > 0 else 0.5
        new = "B" if np.random.rand() < mu_AB else "A" if np.random.rand() < prob_A else ("A" if np.random.rand() < mu_BA else "B")
        i += 1 if new == "A" and i < N else -1 if new == "B" and i > 0 else 0
        history.append(p_A)
    return history

# Run replicate simulations