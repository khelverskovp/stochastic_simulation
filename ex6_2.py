import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Given constants
A1 = 4
A2 = 4
m = 10

# Define the joint distribution P(i, j)
def P(i, j):
    if 0 <= i + j <= m:
        return (A1**i / np.math.factorial(i)) * (A2**j / np.math.factorial(j))
    else:
        return 0

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations):
    # Initial state
    i, j = 0, 0
    samples = [(i, j)]

    for _ in range(iterations):
        # Propose new state
        i_new = np.random.randint(0, m + 1)
        j_new = np.random.randint(0, m + 1 - i_new)
        
        # Calculate acceptance ratio
        alpha = min(1, P(i_new, j_new) / P(i, j))
        
        # Accept or reject the new state
        if np.random.rand() < alpha:
            i, j = i_new, j_new
            
        samples.append((i, j))
    
    return samples

# Coordinate-wise Metropolis-Hastings algorithm
def coordinate_wise_metropolis_hastings(iterations):
    # Initial state
    i, j = 0, 0
    samples = [(i, j)]

    for _ in range(iterations):
        # Update i
        i_new = np.random.randint(0, m + 1)
        alpha_i = min(1, P(i_new, j) / P(i, j))
        if np.random.rand() < alpha_i:
            i = i_new
        
        # Update j
        j_new = np.random.randint(0, m + 1 - i)
        alpha_j = min(1, P(i, j_new) / P(i, j))
        if np.random.rand() < alpha_j:
            j = j_new
        
        samples.append((i, j))
    
    return samples

# Gibbs sampling algorithm
def gibbs_sampling(iterations):
    # Initial state
    i, j = 0, 0
    samples = [(i, j)]

    for _ in range(iterations):
        # Sample i given j
        i_probs = np.array([P(k, j) for k in range(m + 1 - j)])
        i_probs /= i_probs.sum()
        i = np.random.choice(range(m + 1 - j), p=i_probs)
        
        # Sample j given i
        j_probs = np.array([P(i, k) for k in range(m + 1 - i)])
        j_probs /= j_probs.sum()
        j = np.random.choice(range(m + 1 - i), p=j_probs)
        
        samples.append((i, j))
    
    return samples

# Generate samples
iterations = 10000
samples_mh = metropolis_hastings(iterations)
samples_cmh = coordinate_wise_metropolis_hastings(iterations)
samples_gibbs = gibbs_sampling(iterations)

# Plotting the samples
def plot_samples(samples, title):
    i_samples = [sample[0] for sample in samples]
    j_samples = [sample[1] for sample in samples]

    plt.hist2d(i_samples, j_samples, bins=(m + 1, m + 1), cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel('i')
    plt.ylabel('j')
    plt.title(title)
    plt.show()

plot_samples(samples_mh, 'Metropolis-Hastings Sampling')
plot_samples(samples_cmh, 'Coordinate-wise Metropolis-Hastings Sampling')
plot_samples(samples_gibbs, 'Gibbs Sampling')

# Chi-Square Test
# Calculate expected frequencies
expected_freq = np.zeros((m + 1, m + 1))
total_sum = sum(P(i, j) for i in range(m + 1) for j in range(m + 1 - i))
for i in range(m + 1):
    for j in range(m + 1 - i):
        expected_freq[i, j] = P(i, j) / total_sum

# Calculate observed frequencies from samples
def get_observed_freq(samples):
    observed_freq = np.zeros((m + 1, m + 1))
    for i, j in samples:
        observed_freq[i, j] += 1
    return observed_freq / len(samples)

observed_freq_mh = get_observed_freq(samples_mh)
observed_freq_cmh = get_observed_freq(samples_cmh)
observed_freq_gibbs = get_observed_freq(samples_gibbs)

# Flatten the frequencies and remove zero expected frequencies
def flatten_and_filter(expected, observed):
    expected_flat = expected.flatten()
    observed_flat = observed.flatten()
    nonzero_indices = expected_flat > 0
    return expected_flat[nonzero_indices], observed_flat[nonzero_indices]

expected_freq_flat, observed_freq_mh_flat = flatten_and_filter(expected_freq, observed_freq_mh)
_, observed_freq_cmh_flat = flatten_and_filter(expected_freq, observed_freq_cmh)
_, observed_freq_gibbs_flat = flatten_and_filter(expected_freq, observed_freq_gibbs)

# Chi-square test
chi2_mh, p_mh = chisquare(observed_freq_mh_flat, f_exp=expected_freq_flat)
chi2_cmh, p_cmh = chisquare(observed_freq_cmh_flat, f_exp=expected_freq_flat)
chi2_gibbs, p_gibbs = chisquare(observed_freq_gibbs_flat, f_exp=expected_freq_flat)

print(f"Metropolis-Hastings: Chi-square = {chi2_mh}, p-value = {p_mh}")
print(f"Coordinate-wise Metropolis-Hastings: Chi-square = {chi2_cmh}, p-value = {p_cmh}")
print(f"Gibbs Sampling: Chi-square = {chi2_gibbs}, p-value = {p_gibbs}")
