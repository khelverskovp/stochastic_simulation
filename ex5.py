import numpy as np
import scipy.stats as stats
import pandas as pd

# Function to estimate the integral using Crude Monte Carlo
def crude_monte_carlo(n_samples=100):
    samples = np.random.uniform(0, 1, n_samples)
    estimates = np.exp(samples)
    point_estimate = np.mean(estimates)
    variance = np.var(estimates, ddof=1)
    standard_error = np.sqrt(variance / n_samples)
    confidence_interval = stats.norm.interval(0.95, loc=point_estimate, scale=standard_error)
    return point_estimate, confidence_interval

# Function to estimate the integral using Antithetic Variables
def antithetic_variables(n_samples=100):
    u = np.random.uniform(0, 1, n_samples//2)
    estimates = np.exp(u)
    antithetic_estimates = np.exp(1 - u)
    combined_estimates = (estimates + antithetic_estimates) / 2
    point_estimate = np.mean(combined_estimates)
    variance = np.var(combined_estimates, ddof=1)
    standard_error = np.sqrt(variance / (n_samples//2))
    confidence_interval = stats.norm.interval(0.95, loc=point_estimate, scale=standard_error)
    return point_estimate, confidence_interval

# Function to estimate the integral using Control Variable
def control_variable(n_samples=100):
    u = np.random.uniform(0, 1, n_samples)
    estimates = np.exp(u)
    control_var = u
    control_mean = 0.5
    cov = np.cov(estimates, control_var)[0, 1]
    var_control = np.var(control_var, ddof=1)
    c = -cov / var_control
    adjusted_estimates = estimates + c * (control_var - control_mean)
    point_estimate = np.mean(adjusted_estimates)
    variance = np.var(adjusted_estimates, ddof=1)
    standard_error = np.sqrt(variance / n_samples)
    confidence_interval = stats.norm.interval(0.95, loc=point_estimate, scale=standard_error)
    return point_estimate, confidence_interval

# Function to estimate the integral using Stratified Sampling
def stratified_sampling(n_samples=100):
    strata = np.linspace(0, 1, n_samples + 1)
    estimates = []
    for i in range(n_samples):
        u = np.random.uniform(strata[i], strata[i+1])
        estimates.append(np.exp(u))
    estimates = np.array(estimates)
    point_estimate = np.mean(estimates)
    variance = np.var(estimates, ddof=1)
    standard_error = np.sqrt(variance / n_samples)
    confidence_interval = stats.norm.interval(0.95, loc=point_estimate, scale=standard_error)
    return point_estimate, confidence_interval

# Estimation results
n_samples = 100
results = {
    'Method': ['Crude Monte Carlo', 'Antithetic Variables', 'Control Variable', 'Stratified Sampling'],
    'Point Estimate': [
        crude_monte_carlo(n_samples)[0],
        antithetic_variables(n_samples)[0],
        control_variable(n_samples)[0],
        stratified_sampling(n_samples)[0]
    ],
    'Confidence Interval Lower': [
        crude_monte_carlo(n_samples)[1][0],
        antithetic_variables(n_samples)[1][0],
        control_variable(n_samples)[1][0],
        stratified_sampling(n_samples)[1][0]
    ],
    'Confidence Interval Upper': [
        crude_monte_carlo(n_samples)[1][1],
        antithetic_variables(n_samples)[1][1],
        control_variable(n_samples)[1][1],
        stratified_sampling(n_samples)[1][1]
    ]
}

results_df = pd.DataFrame(results)

print(results_df)

# part 7
def crude_monte_carlo(a, sample_size):
    # Generate samples from N(0,1)
    samples = np.random.normal(0, 1, sample_size)
    # Estimate the probability
    prob_estimate = np.mean(samples > a)
    return prob_estimate

def importance_sampling(a, sigma_squared, sample_size):
    # Generate samples from N(a, sigma^2)
    samples = np.random.normal(a, np.sqrt(sigma_squared), sample_size)
    # Calculate weights
    weights = np.exp(-0.5 * (samples ** 2 - ((samples - a) ** 2) / sigma_squared))
    # Estimate the probability
    prob_estimate = np.mean(weights * (samples > a))
    return prob_estimate

# Parameters
a_values = [2, 4]
sample_sizes = [1000, 10000, 100000]
sigma_squared = 1

# Running simulations
results = {}
for a in a_values:
    results[a] = {}
    for sample_size in sample_sizes:
        crude_estimate = crude_monte_carlo(a, sample_size)
        importance_estimate = importance_sampling(a, sigma_squared, sample_size)
        results[a][sample_size] = {
            "crude_monte_carlo": crude_estimate,
            "importance_sampling": importance_estimate
        }

# Display results
for a in a_values:
    print(f"Results for a = {a}")
    print("Sample Size\tCrude Monte Carlo\tImportance Sampling")
    for sample_size in sample_sizes:
        crude_estimate = results[a][sample_size]["crude_monte_carlo"]
        importance_estimate = results[a][sample_size]["importance_sampling"]
        print(f"{sample_size}\t{crude_estimate}\t{importance_estimate}")
    print()

# part 8
# Target function
def f(x):
    return np.exp(x)

# Proposal distribution g(x) with parameter lambda
def g(x, lam):
    return lam * np.exp(-lam * x)

# Generate samples from the exponential distribution
def sample_g(lam, size):
    return np.random.exponential(scale=1/lam, size=size)

# Importance sampling estimator
def importance_sampling_estimator(lam, sample_size):
    samples = sample_g(lam, sample_size)
    # Only consider samples in [0,1]
    samples = samples[samples <= 1]
    weights = f(samples) / g(samples, lam)
    return np.mean(weights)

# Variance calculation for the importance sampling estimator
def importance_sampling_variance(lam, sample_size):
    samples = sample_g(lam, sample_size)
    # Only consider samples in [0,1]
    samples = samples[samples <= 1]
    weights = f(samples) / g(samples, lam)
    return np.var(weights)

# Parameters
lambdas = np.linspace(0.1, 10, 100)
sample_size = 10000

# Find the optimal lambda by minimizing the variance
variances = [importance_sampling_variance(lam, sample_size) for lam in lambdas]
optimal_lambda = lambdas[np.argmin(variances)]

# Verify by simulation
estimated_integral = importance_sampling_estimator(optimal_lambda, sample_size)

print(f"Optimal lambda: {optimal_lambda}")
print(f"Estimated integral: {estimated_integral}")

