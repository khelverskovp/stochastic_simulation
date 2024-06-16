import numpy as np
import matplotlib.pyplot as plt

# Define a function to generate geometric distribution samples
def simulate_geometric(p, size=10000):
    return np.random.geometric(p, size)

# Simulate for different values of p
p_values = [0.1, 0.5, 0.9]
samples = {p: simulate_geometric(p) for p in p_values}

# Plot the histograms
plt.figure(figsize=(18, 5))
for i, p in enumerate(p_values):
    plt.subplot(1, 3, i + 1)
    plt.hist(samples[p], bins=range(1, max(samples[p]) + 1), density=True, alpha=0.75, edgecolor='black')
    plt.title(f'Geometric Distribution (p={p})')
    plt.xlabel('Number of Trials')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Define the probabilities
probabilities = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
values = [1, 2, 3, 4, 5, 6]

# Direct method
def simulate_direct(probabilities, values, size=10000):
    return np.random.choice(values, size=size, p=probabilities)

# Generate samples
direct_samples = simulate_direct(probabilities, values)

# Plot the histogram
plt.hist(direct_samples, bins=np.arange(1, 8) - 0.5, density=True, alpha=0.75, edgecolor='black')
plt.title('6-Point Distribution (Direct Method)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks(values)
plt.show()

def simulate_rejection(probabilities, values, size=10000):
    max_p = max(probabilities)
    samples = []
    while len(samples) < size:
        x = np.random.choice(values)
        u = np.random.uniform(0, max_p)
        if u < probabilities[values.index(x)]:
            samples.append(x)
    return samples

# Generate samples
rejection_samples = simulate_rejection(probabilities, values)

# Plot the histogram
plt.hist(rejection_samples, bins=np.arange(1, 8) - 0.5, density=True, alpha=0.75, edgecolor='black')
plt.title('6-Point Distribution (Rejection Method)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks(values)
plt.show()

class AliasMethod:
    def __init__(self, probabilities):
        self.n = len(probabilities)
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=int)
        
        scaled_prob = np.array(probabilities) * self.n
        small = []
        large = []
        
        for i, prob in enumerate(scaled_prob):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
                
        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            
            self.prob[small_index] = scaled_prob[small_index]
            self.alias[small_index] = large_index
            
            scaled_prob[large_index] = scaled_prob[large_index] + scaled_prob[small_index] - 1.0
            
            if scaled_prob[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)
                
        while large:
            self.prob[large.pop()] = 1.0
        while small:
            self.prob[small.pop()] = 1.0
            
    def sample(self):
        i = np.random.randint(0, self.n)
        if np.random.rand() < self.prob[i]:
            return i
        else:
            return self.alias[i]
        
    def generate_samples(self, size=10000):
        return [self.sample() + 1 for _ in range(size)]

# Create an instance of the AliasMethod class
alias_method = AliasMethod(probabilities)

# Generate samples
alias_samples = alias_method.generate_samples()

# Plot the histogram
plt.hist(alias_samples, bins=np.arange(1, 8) - 0.5, density=True, alpha=0.75, edgecolor='black')
plt.title('6-Point Distribution (Alias Method)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks(values)
plt.show()

from scipy.stats import chisquare

# Function to perform chi-square test
def chi_square_test(samples, probabilities):
    observed_freq, _ = np.histogram(samples, bins=np.arange(1, 8) - 0.5)
    expected_freq = np.array(probabilities) * len(samples)
    chi2, p_value = chisquare(observed_freq, expected_freq)
    return chi2, p_value

# Perform chi-square tests
direct_chi2, direct_p = chi_square_test(direct_samples, probabilities)
rejection_chi2, rejection_p = chi_square_test(rejection_samples, probabilities)
alias_chi2, alias_p = chi_square_test(alias_samples, probabilities)

# Print the results
print("Chi-square test results:")
print(f"Direct Method: chi2 = {direct_chi2:.2f}, p-value = {direct_p:.4f}")
print(f"Rejection Method: chi2 = {rejection_chi2:.2f}, p-value = {rejection_p:.4f}")
print(f"Alias Method: chi2 = {alias_chi2:.2f}, p-value = {alias_p:.4f}")

# Calculate the observed frequencies
def get_observed_frequencies(samples, values):
    observed_freq, _ = np.histogram(samples, bins=np.arange(1, 8) - 0.5)
    return observed_freq

# Get observed frequencies for each method
direct_freq = get_observed_frequencies(direct_samples, values)
rejection_freq = get_observed_frequencies(rejection_samples, values)
alias_freq = get_observed_frequencies(alias_samples, values)

# Print observed frequencies
print("Observed Frequencies:")
print(f"Direct Method: {direct_freq}")
print(f"Rejection Method: {rejection_freq}")
print(f"Alias Method: {alias_freq}")

# Verify if frequencies are exactly the same
print("Frequencies are the same for all methods:", 
      np.array_equal(direct_freq, rejection_freq) and np.array_equal(direct_freq, alias_freq))
