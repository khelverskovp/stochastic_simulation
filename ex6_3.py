# part a
import numpy as np

# Define the mean vector and covariance matrix
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])

# Generate a sample (xi, gamma) from the bivariate normal distribution
xi, gamma = np.random.multivariate_normal(mean, cov)

# Compute theta and psi
theta = np.exp(xi)
psi = np.exp(gamma)

print(f"Sample (xi, gamma): ({xi}, {gamma})")
print(f"Generated pair (theta, psi): ({theta}, {psi})")

# part b
n = 10

X_i = np.random.normal(loc=theta, scale=np.sqrt(psi), size=n)

print(f"Generated sample X_i: {X_i}")

# part d
# Define the posterior density function (up to a proportionality constant)
def posterior_density(theta, psi, sample_mean, sample_variance, n):
    if theta <= 0 or psi <= 0:
        return 0
    
    prior_density = 1 / (2 * np.pi * theta * psi * np.sqrt(1 - 0.5**2)) * \
                    np.exp(- (np.log(theta)**2 - 2 * 0.5 * np.log(theta) * np.log(psi) + np.log(psi)**2) / (2 * (1 - 0.5**2)))
    
    if np.isnan(prior_density) or prior_density <= 0:
        return 0

    likelihood = (psi**(-n/2 - 1)) * np.exp(-0.5 * ((n - 1) * sample_variance + n * (sample_mean - theta)**2) / psi)
    
    if np.isnan(likelihood) or likelihood <= 0:
        return 0

    return likelihood * prior_density

# Generate Metropolis-Hastings samples
def metropolis_hastings(n_samples, initial_theta, initial_psi, sample_mean, sample_variance, n):
    samples = []
    theta_current = initial_theta
    psi_current = initial_psi

    for _ in range(n_samples):
        # Propose new state using a log-normal proposal to ensure positivity
        theta_proposal = np.random.lognormal(np.log(theta_current), 0.1)
        psi_proposal = np.random.lognormal(np.log(psi_current), 0.1)
        
        # Compute acceptance ratio
        posterior_current = posterior_density(theta_current, psi_current, sample_mean, sample_variance, n)
        posterior_proposal = posterior_density(theta_proposal, psi_proposal, sample_mean, sample_variance, n)
        alpha = min(1, posterior_proposal / posterior_current)
        
        # Accept or reject
        if np.random.rand() < alpha:
            theta_current = theta_proposal
            psi_current = psi_proposal
        
        samples.append((theta_current, psi_current))
    
    return np.array(samples)

# Given sample statistics from part (b)
sample_mean = np.mean(X_i)
sample_variance = np.var(X_i, ddof=1)

# Initial values for theta and psi
initial_theta = np.mean(X_i)
initial_psi = np.var(X_i, ddof=1)

# Number of samples to generate
n_samples = 10000

# Generate MCMC samples
mcmc_samples = metropolis_hastings(n_samples, initial_theta, initial_psi, sample_mean, sample_variance, len(X_i))

print(mcmc_samples)

# visualizing the samples
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(mcmc_samples[:, 0], label='Theta')
plt.plot(mcmc_samples[:, 1], label='Psi')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('MCMC Samples')
plt.legend()
plt.show()

# part e
# Values of theta and psi from part (a)
theta = 1.2  # Replace with actual theta value from part (a)
psi = 0.8    # Replace with actual psi value from part (a)

# Number of samples for X_i
n_100 = 100
n_1000 = 1000

# Generate samples X_i for n = 100 and n = 1000
X_i_100 = np.random.normal(loc=theta, scale=np.sqrt(psi), size=n_100)
X_i_1000 = np.random.normal(loc=theta, scale=np.sqrt(psi), size=n_1000)

# Compute sample statistics
sample_mean_100 = np.mean(X_i_100)
sample_variance_100 = np.var(X_i_100, ddof=1)
sample_mean_1000 = np.mean(X_i_1000)
sample_variance_1000 = np.var(X_i_1000, ddof=1)

# Initial values for theta and psi
initial_theta = np.mean(X_i_100)
initial_psi = np.var(X_i_100, ddof=1)

# Number of MCMC samples to generate
n_mcmc_samples = 10000

# Generate MCMC samples for n=100 and n=1000
mcmc_samples_100 = metropolis_hastings(n_mcmc_samples, initial_theta, initial_psi, sample_mean_100, sample_variance_100, n_100)
mcmc_samples_1000 = metropolis_hastings(n_mcmc_samples, initial_theta, initial_psi, sample_mean_1000, sample_variance_1000, n_1000)

# Plot the samples
plt.figure(figsize=(10, 6))
plt.plot(mcmc_samples_100[:, 0], label='Theta (n=100)')
plt.plot(mcmc_samples_100[:, 1], label='Psi (n=100)')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('MCMC Samples for n=100')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(mcmc_samples_1000[:, 0], label='Theta (n=1000)')
plt.plot(mcmc_samples_1000[:, 1], label='Psi (n=1000)')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('MCMC Samples for n=1000')
plt.legend()
plt.show()