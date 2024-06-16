import numpy as np

# Exercise 13 Ross
# Given data
X = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
a, b = -5, 5
n = len(X)
mu = np.mean(X)
B = 10000  # Number of bootstrap samples

# Bootstrap resampling
bootstrap_samples = np.random.choice(X, (B, n), replace=True)
bootstrap_means = np.mean(bootstrap_samples, axis=1)

# Calculate the statistic for each bootstrap sample
statistic = bootstrap_means - mu

# Estimate p
p_estimate = np.mean((statistic > a) & (statistic < b))

print(p_estimate)


# Exercise 15 Ross
# Given data
data = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])
n = len(data)
B = 10000  # Number of bootstrap samples

# Function to calculate sample variance
def sample_variance(x):
    return np.var(x, ddof=1)

# Calculate original sample variance
original_sample_variance = sample_variance(data)

# Bootstrap resampling
bootstrap_samples = np.random.choice(data, (B, n), replace=True)
bootstrap_variances = np.array([sample_variance(sample) for sample in bootstrap_samples])

# Estimate Var(S^2)
var_S2_estimate = np.var(bootstrap_variances, ddof=1)

print(var_S2_estimate)