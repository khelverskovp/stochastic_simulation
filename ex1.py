import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2

# Implementing LCG function
def lcg(a, c, m, x0, n):
    random_numbers = []
    for i in range(n):
        x0 = (a * x0 + c) % m
        random_numbers.append(x0)
    return random_numbers

# Good choice of parameters
good_a = 1664525
good_c = 1013904223
good_m = 2**32
good_x0 = 3 # Seed
good_numbers = lcg(good_a, good_c, good_m, good_x0, 10000)

# Bad choice of parameters
bad_a = 5
bad_c = 1  
bad_m = 16  
bad_x0 = 3  
bad_numbers = lcg(bad_a, bad_c, bad_m, bad_x0, 10000)

# System-available generator (Python built-in random module)
random.seed(3)  # Seed
system_numbers = [random.randint(0, good_m - 1) for _ in range(10000)]

# Plotting histograms
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(good_numbers, bins=30, color='blue', alpha=0.5)
plt.title('Good Parameters')
plt.xlabel('Generated Number')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(bad_numbers, bins=30, color='red', alpha=0.5)
plt.title('Bad Parameters')
plt.xlabel('Generated Number')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(system_numbers, bins=30, color='green', alpha=0.5)
plt.title('System Generator')
plt.xlabel('Generated Number')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

def chi2_test(n_observed, n_expected):
    return sum((n_observed - n_expected)**2 / n_expected)

# Chi-squared test
n_classes = len(set(good_numbers))
n_observed, _ = np.histogram(good_numbers, bins=n_classes)
n_expected = np.full(n_classes, 10000 / n_classes)
chi2_value = chi2_test(n_observed, n_expected)
print('Good Parameters:')
print('Chi-squared value:', chi2_value)
print('p-value:', 1 - stats.chi2.cdf(chi2_value, n_classes - 1))

n_classes = len(set(bad_numbers))
n_observed, _ = np.histogram(bad_numbers, bins=n_classes)
n_expected = np.full(n_classes, 10000 / n_classes)
chi2_value = chi2_test(n_observed, n_expected)
print('Bad Parameters:')
print('Chi-squared value:', chi2_value)
print('p-value:', 1 - stats.chi2.cdf(chi2_value, n_classes - 1))

n_classes = len(set(system_numbers))
n_observed, _ = np.histogram(system_numbers, bins=n_classes)
n_expected = np.full(n_classes, 10000 / n_classes)
chi2_value = chi2_test(n_observed, n_expected)
print('System Generator:')
print('Chi-squared value:', chi2_value)
print('p-value:', 1 - stats.chi2.cdf(chi2_value, n_classes - 1))

# Kolmogorov-Smirnov test
def kolmogorov_smirnov_test(numbers):
    n = len(numbers)
    sorted_numbers = np.sort(numbers)

    D_plus = np.max((np.arange(1, n + 1) / n) - sorted_numbers)
    D_minus = np.max(sorted_numbers - (np.arange(0, n) / n))
    
    D = max(D_plus, D_minus)
    return D

# Normalize the good, bad, and system numbers for comparison
good_numbers_normalized = [x / good_m for x in good_numbers]
bad_numbers_normalized = [x / bad_m for x in bad_numbers]
system_numbers_normalized = [x / good_m for x in system_numbers]

ks_good = kolmogorov_smirnov_test(good_numbers_normalized)
ks_bad = kolmogorov_smirnov_test(bad_numbers_normalized)
ks_system = kolmogorov_smirnov_test(system_numbers_normalized)

ks_good, ks_bad, ks_system

# p-value from the Kolmogorov-Smirnov distribution
n = 10000
p_value_good = 1 - stats.kstwobign.cdf(ks_good * np.sqrt(n))
p_value_bad = 1 - stats.kstwobign.cdf(ks_bad * np.sqrt(n))
p_value_system = 1 - stats.kstwobign.cdf(ks_system * np.sqrt(n))

# print results
print('Good Parameters:')
print('Kolmogorov-Smirnov test:', ks_good)
print('p-value:', p_value_good)

print('Bad Parameters:')
print('Kolmogorov-Smirnov test:', ks_bad)
print('p-value:', p_value_bad)

print('System Generator:')
print('Kolmogorov-Smirnov test:', ks_system)
print('p-value:', p_value_system)

# scatter plots of x_i vs x_{i+1}
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(good_numbers[:-1], good_numbers[1:], color='blue', alpha=0.5)
plt.title('Good Parameters')
plt.xlabel('$x_i$')
plt.ylabel('$x_{i+1}$')

plt.subplot(1, 3, 2)
plt.scatter(bad_numbers[:-1], bad_numbers[1:], color='red', alpha=0.5)
plt.title('Bad Parameters')
plt.xlabel('$x_i$')
plt.ylabel('$x_{i+1}$')

plt.subplot(1, 3, 3)
plt.scatter(system_numbers[:-1], system_numbers[1:], color='green', alpha=0.5)
plt.title('System Generator')
plt.xlabel('$x_i$')
plt.ylabel('$x_{i+1}$')

plt.tight_layout()
plt.show()

def WW_test(sequence):
    # Calculate the median
    median = np.median(sequence)
    
    # Convert sequence to a sequence of 'above' (1) and 'below' (0)
    runs_sequence = ['above' if x > median else 'below' for x in sequence]
    
    # Calculate the number of runs
    runs = 1  # There's at least one run
    for i in range(1, len(runs_sequence)):
        if runs_sequence[i] != runs_sequence[i-1]:
            runs += 1
    
    # Calculate n1 and n2
    n1 = runs_sequence.count('above')
    n2 = runs_sequence.count('below')
    
    # Calculate the mean and standard deviation
    mean = (2 * n1 * n2) / (n1 + n2) + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    std_dev = np.sqrt(variance)
    
    # Calculate the z-score
    z = (runs - mean) / std_dev
    
    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    
    return runs, z, p_value

# Perform the runs test on the three sequences
good_runs, good_z, good_p = WW_test(good_numbers)
bad_runs, bad_z, bad_p = WW_test(bad_numbers)
system_runs, system_z, system_p = WW_test(system_numbers)

# Print the results
print("Good LCG:")
print(f"Number of runs: {good_runs}, Z-score: {good_z}, P-value: {good_p}")

print("\nBad LCG:")
print(f"Number of runs: {bad_runs}, Z-score: {bad_z}, P-value: {bad_p}")

print("\nSystem Generator:")
print(f"Number of runs: {system_runs}, Z-score: {system_z}, P-value: {system_p}")

# Compute estimated correlation c_h for a given lag h
def compute_correlation(sequence, h):
    n = len(sequence)
    correlation = np.sum([sequence[i] * sequence[i + h] for i in range(n - h)]) / (n - h)
    return correlation

# Test the correlation for various lags
lags = [1, 2, 5, 10]  # Example lags
correlations = []
for h in lags:
    good_correlation = compute_correlation(good_numbers, h)
    bad_correlation = compute_correlation(bad_numbers, h)
    system_correlation = compute_correlation(system_numbers, h)
    correlations.append((good_correlation, bad_correlation, system_correlation))

# Print the results
for i, h in enumerate(lags):
    print(f"Lag {h}:")
    print(f"Good LCG: {correlations[i][0]}")
    print(f"Bad LCG: {correlations[i][1]}")
    print(f"System Generator: {correlations[i][2]}")
    print()

# plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(lags, [c[0] for c in correlations], marker='o', linestyle='-', color='blue', label='Good Parameters')
plt.title('Good Parameters')
plt.xlabel('Lag')
plt.ylabel('Estimated Correlation')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(lags, [c[1] for c in correlations], marker='o', linestyle='-', color='red', label='Bad Parameters')
plt.title('Bad Parameters')
plt.xlabel('Lag')
plt.ylabel('Estimated Correlation')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(lags, [c[2] for c in correlations], marker='o', linestyle='-', color='green', label='System Generator')
plt.title('System Generator')
plt.xlabel('Lag')
plt.ylabel('Estimated Correlation')
plt.legend()

plt.tight_layout()
plt.show()