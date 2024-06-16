import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define parameters
A = 5  
m = 10

# Compute normalization constant c
def compute_normalization_constant(A, m):
    c = sum((A ** i) / np.math.factorial(i) for i in range(m + 1))
    return 1 / c

c = compute_normalization_constant(A, m)

# Define the truncated Poisson distribution
def truncated_poisson_pmf(i, A, m, c):
    if 0 <= i <= m:
        return c * (A ** i) / np.math.factorial(i)
    else:
        return 0

# Metropolis-Hastings algorithm
def metropolis_hastings(A, m, c, num_samples=10000):
    samples = []
    current = np.random.randint(0, m + 1)  # Initial state

    for _ in range(num_samples):
        proposal = np.random.randint(0, m + 1)  # Randomly propose a new state
        acceptance_ratio = truncated_poisson_pmf(proposal, A, m, c) / truncated_poisson_pmf(current, A, m, c)
        
        if np.random.rand() < acceptance_ratio:
            current = proposal
        
        samples.append(current)
    
    return np.array(samples)

# Generate samples
samples = metropolis_hastings(A, m, c, num_samples=10000)

# Perform chi-square test
observed_freq, _ = np.histogram(samples, bins=np.arange(m + 2) - 0.5, density=False)
expected_freq = np.array([truncated_poisson_pmf(i, A, m, c) for i in range(m + 1)]) * len(samples)

chi2, p_value = stats.chisquare(observed_freq, f_exp=expected_freq)

# Print the results
print(f"Chi-squared test statistic: {chi2}")
print(f"P-value: {p_value}")

# Plot the observed and expected frequencies
plt.bar(range(m + 1), observed_freq, alpha=0.6, label='Observed')
plt.bar(range(m + 1), expected_freq, alpha=0.6, label='Expected')
plt.xlabel('Number of Busy Lines')
plt.ylabel('Frequency')
plt.legend()
plt.show()
