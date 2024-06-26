import numpy as np

def generate_arrival_times(n_customers, a_type, uniform_random_numbers=None):
    if a_type == 'poisson':
        if uniform_random_numbers is None:
            arrival_times = np.random.exponential(mean_bc, n_customers)
        else:
            arrival_times = -np.log(1 - uniform_random_numbers) * mean_bc
    elif a_type == 'hyperexponential':
        p1 = 0.8
        lambda1 = 1 / 1.2
        lambda2 = 1 / 5
        if uniform_random_numbers is None:
            arrival_times = np.random.exponential(1 / lambda1, n_customers)
            for i in range(n_customers):
                if np.random.uniform() < p1:
                    arrival_times[i] = np.random.exponential(1 / lambda1)
                else:
                    arrival_times[i] = np.random.exponential(1 / lambda2)
        else:
            arrival_times = np.where(uniform_random_numbers < p1,
                                     -np.log(1 - uniform_random_numbers) / lambda1,
                                     -np.log(1 - uniform_random_numbers) / lambda2)
    return arrival_times

def generate_service_times(n_customers, s_type, uniform_random_numbers=None):
    if s_type == 'exponential':
        if uniform_random_numbers is None:
            service_times = np.random.exponential(mean_st, n_customers)
        else:
            service_times = -np.log(1 - uniform_random_numbers) * mean_st
    return service_times

def simulation(m, mean_st, n_customers, arrival_times, service_times):
    n_blocked_customers = 0
    service_units = np.zeros(m)

    time = 0
    for i in range(n_customers):
        time += arrival_times[i]
        service_units = np.maximum(0, service_units - arrival_times[i])

        if np.sum(service_units == 0) > 0:
            service_units[np.where(service_units == 0)[0][0]] = service_times[i]
        else:
            n_blocked_customers += 1

    return n_blocked_customers

def run_simulations(m, mean_st, mean_bc, n_customers, s_type, num_simulations=10, use_crn=False):
    poisson_fractions = []
    hyperexponential_fractions = []

    for _ in range(num_simulations):
        if use_crn:
            uniform_random_numbers = np.random.uniform(size=n_customers)
        else:
            uniform_random_numbers = None

        poisson_arrival_times = generate_arrival_times(n_customers, 'poisson', uniform_random_numbers)
        hyperexponential_arrival_times = generate_arrival_times(n_customers, 'hyperexponential', uniform_random_numbers)
        service_times = generate_service_times(n_customers, s_type, uniform_random_numbers)

        poisson_blocked = simulation(m, mean_st, n_customers, poisson_arrival_times, service_times)
        hyperexponential_blocked = simulation(m, mean_st, n_customers, hyperexponential_arrival_times, service_times)

        poisson_fraction = poisson_blocked / n_customers
        hyperexponential_fraction = hyperexponential_blocked / n_customers

        poisson_fractions.append(poisson_fraction)
        hyperexponential_fractions.append(hyperexponential_fraction)

    differences = np.array(poisson_fractions) - np.array(hyperexponential_fractions)
    variance_of_differences = np.var(differences)

    return variance_of_differences, poisson_fractions, hyperexponential_fractions

# Parameters
m = 10
mean_st = 8
mean_bc = 1
n_customers = 10 * 10000
s_type = 'exponential'
num_simulations = 10

# Run simulations without CRN
variance_without_crn, poisson_fractions_without_crn, hyperexponential_fractions_without_crn = run_simulations(m, mean_st, mean_bc, n_customers, s_type, num_simulations, use_crn=False)

# Run simulations with CRN
variance_with_crn, poisson_fractions_with_crn, hyperexponential_fractions_with_crn = run_simulations(m, mean_st, mean_bc, n_customers, s_type, num_simulations, use_crn=True)

# Output results
print('Variance of differences without CRN:', variance_without_crn)
print('Variance of differences with CRN:', variance_with_crn)
print('Reduction in variance due to CRN:', variance_without_crn - variance_with_crn)
print('Poisson arrival fractions without CRN:', poisson_fractions_without_crn)
print('Hyperexponential arrival fractions without CRN:', hyperexponential_fractions_without_crn)
print('Poisson arrival fractions with CRN:', poisson_fractions_with_crn)
print('Hyperexponential arrival fractions with CRN:', hyperexponential_fractions_with_crn)

