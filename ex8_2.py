import numpy as np

def bootstrap_variance(data, statistic, B=100):
    """Calculate the bootstrap estimate of the variance of a statistic."""
    n = len(data)
    bootstrap_samples = np.random.choice(data, (B, n), replace=True)
    bootstrap_statistics = np.array([statistic(sample) for sample in bootstrap_samples])
    return np.var(bootstrap_statistics, ddof=1)

# Parameters for Pareto distribution
N = 200
beta = 1
k = 1.05

# Generate Pareto distributed random variates
data = (np.random.pareto(k, N) + 1) * beta

# (a) Compute the mean and the median
mean_sample = np.mean(data)
median_sample = np.median(data)

# (b) Bootstrap estimate of the variance of the sample mean
variance_mean = bootstrap_variance(data, np.mean, B=100)

# (c) Bootstrap estimate of the variance of the sample median
variance_median = bootstrap_variance(data, np.median, B=100)

# (d) Compare the precision
precision_comparison = {
    "mean_sample": mean_sample,
    "median_sample": median_sample,
    "variance_mean": variance_mean,
    "variance_median": variance_median
}

import pandas as pd

# Convert the dictionary to a DataFrame for better visualization
precision_df = pd.DataFrame(precision_comparison, index=[0])

print(precision_df)
