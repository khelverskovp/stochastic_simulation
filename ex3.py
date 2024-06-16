import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# exponential distribution
def exp(lambda_, U_n=10000):
	U = np.random.uniform(size=U_n)
	return -np.log(U) / lambda_

lambda_ = 1
X = exp(lambda_)

plt.hist(X,100)
plt.title('Histogram of Exponential Distribution')
plt.show()

# Kolmogorov-Smirnov test for exponential distribution
print("ks-test for exponential distribution")
D, p = stats.kstest(X, 'expon', args=(0, lambda_))
print('KS test statistic: ', D)
print('p-value: ', p)

# computed mean and variance
print('Computed Mean: ', np.mean(X))
print('Computed Variance: ', np.var(X))

# analytical mean and variance
mean = 1/lambda_
variance = 1/(lambda_**2)
print('Analytical Mean: ', mean)
print('Analytical Variance: ', variance)

# standard normal distribution Box-Muller method
def normal(U1, U2):
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return Z1, Z2

U1 = np.random.uniform(size=10000)
U2 = np.random.uniform(size=10000)
Z1, Z2 = normal(U1, U2)

plt.hist(Z1,100)
plt.title('Histogram of Standard Normal Distribution')
plt.show()

#kolmogorov-smirnov test for normal distribution
print("ks-test for standard normal distribution")
D, p = stats.kstest(Z1, 'norm')
print('KS test statistic: ', D)
print('p-value: ', p)

# generate 100 95% confidence intervals for the mean of the standard normal distribution, each based on 10 observations from Z1
confidence_intervals = []
for i in range(100):
    sample = np.random.choice(Z1, 10)
    mean = np.mean(sample)
    std = np.std(sample)
    confidence_intervals.append((mean - 1.96 * std / np.sqrt(10), mean + 1.96 * std / np.sqrt(10)))

# generate 100 95% confidence intervals for the variance of the standard normal distribution, each based on 10 observations from Z1
confidence_intervals_var = []
for i in range(100):
    sample = np.random.choice(Z1, 10)
    var = np.var(sample)
    confidence_intervals_var.append((var * 9 / 16.92, var * 9 / 2.16))

# print the mean of the confidence intervals for the mean and variance
print('Mean of confidence intervals for the mean: ', np.mean(confidence_intervals))
print('Mean of confidence intervals for the variance: ', np.mean(confidence_intervals_var))


# pareto distribution
def pareto(beta, k, U_n=10000):
    U = np.random.uniform(size=U_n)
    return beta*(np.power(U,-1/k))

# plot histogram and test for k = 2.05, k=2.5, k=3, k=4

for k in [2.05, 2.5, 3, 4]:
    X = pareto(1, k)
    plt.hist(X,100)
    plt.title('Histogram of Pareto Distribution with k = ' + str(k))
    plt.show()

    # Kolmogorov-Smirnov test
    D, p = stats.kstest(X, 'pareto', args=(k,))
    print("ks-test for pareto distribution")
    print('k=', k)
    print('KS test statistic: ', D)
    print('p-value: ', p)

    # analytical mean and variance
    mean = k/(k-1)
    variance = k/((k-1)**2*(k-2))

    # print computed paramaeters and analytical
    print('Computed Mean: ', np.mean(X))
    print('Computed Variance: ', np.var(X))
    print('Analytical Mean: ', mean)
    print('Analytical Variance: ', variance)
    print('\n')


def pareto_samples(k, size=1000):
    # Generate uniform random samples
    U = np.random.uniform(0, 1, size)
    # Transform uniform samples to Pareto distribution
    X = (1 / U) ** (1 / k)
    return X

# Parameters
k_values = [2.05, 2.5, 3, 4]
sample_size = 1000

# Generate and plot Pareto samples for different k values
plt.figure(figsize=(12, 10))
for i, k in enumerate(k_values, 1):
    samples = pareto_samples(k, sample_size)
    plt.subplot(2, 2, i)
    plt.hist(samples, bins=50, density=True, alpha=0.75)
    plt.title(f'Pareto Distribution (k = {k})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
