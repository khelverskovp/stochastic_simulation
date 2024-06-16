import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulation_with_control_variates(m, mean_st, mean_bc, n_customers, a_type, s_type):

    # parameters:
    # m: number of service units
    # mean_st: mean service time
    # mean_bc: mean between customers
    # n_customers: number of customers
    # a_type: arrival type 
    # s_type: service type

    # output: 
    # n_blocked_customers : number of blocked customers

    # initialize variables
    n_blocked_customers = 0

    # initialize service units
    service_units = np.zeros(m)

    # arrival_type_dist 
    if a_type == 'poisson':
        arrival_type_dist = np.random.exponential(mean_bc, n_customers)
    elif a_type == 'erlang':
        arrival_type_dist = np.random.gamma(shape=1, size=n_customers, scale=mean_bc)
    elif a_type == 'hyperexponential':
        p1 = 0.8
        p2 = 0.2
        lambda1 = 1 / 1.2
        lambda2 = 1 / 5
        arrival_type_dist = np.random.exponential(1 / lambda1, n_customers)
        for i in range(n_customers):
            if np.random.uniform() < p1:
                arrival_type_dist[i] = np.random.exponential(1 / lambda1)
            else:
                arrival_type_dist[i] = np.random.exponential(1 / lambda2)

    # service_type_dist 
    if s_type == 'exponential':
        service_type_dist = np.random.exponential(mean_st, n_customers)
    elif s_type == 'constant':
        service_type_dist = np.ones(n_customers) * mean_st
    elif s_type == 'pareto1':
        k = 1.05
        service_type_dist = np.random.pareto(k, n_customers) + 1
    elif s_type == 'pareto2':
        k = 2.05
        service_type_dist = np.random.pareto(k, n_customers) + 1
    elif s_type == 'weibull':
        # Weibull distributed service times
        shape_weibull = 1.5
        scale_weibull = mean_st / np.math.gamma(1 + 1 / shape_weibull)
        service_type_dist = np.random.weibull(shape_weibull, n_customers) * scale_weibull

    # iterate over customers
    time = 0
    total_service_time = 0
    for i in range(n_customers):
        # update time
        time += arrival_type_dist[i]

        # decrement the remaining service times of all occupied service units
        service_units = np.maximum(0, service_units - arrival_type_dist[i])

        # check if there are available service units
        if np.sum(service_units == 0) > 0:
            # assign service unit
            service_units[np.where(service_units == 0)[0][0]] = service_type_dist[i]
        else:
            # block customer
            n_blocked_customers += 1
        
        # Accumulate total service time for control variate
        total_service_time += service_type_dist[i]

    return n_blocked_customers, total_service_time
# Control variate implementation
m = 10
mean_st = 8
mean_bc = 1
n_customers = 10 * 10000
a_type = 'poisson'
s_type = 'exponential'

# run multiple simulations to estimate the expected value of the control variate
num_simulations = 10
blocked_customers = np.zeros(num_simulations)
total_service_times = np.zeros(num_simulations)

for i in range(num_simulations):
    blocked_customers[i], total_service_times[i] = simulation_with_control_variates(m, mean_st, mean_bc, n_customers, a_type, s_type)

# Compute the expected value of the control variate
expected_total_service_time = np.mean(total_service_times)

# Compute control variate adjustment
control_variate_adjustment = (total_service_times - expected_total_service_time) * np.cov(blocked_customers, total_service_times)[0, 1] / np.var(total_service_times)
adjusted_estimates = blocked_customers - control_variate_adjustment

# Calculate adjusted mean
adjusted_estimate_mean = np.mean(adjusted_estimates)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = adjusted_estimate_mean / n_customers
std_error = np.std(adjusted_estimates / n_customers) / np.sqrt(num_simulations)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('poisson arrival time with control variates')
print('Adjusted fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

##### using common random numbers 

def simulation(m, mean_st, interarrival_times, service_times):

    # parameters:
    # m: number of service units
    # mean_st: mean service time
    # interarrival_times: pre-generated interarrival times
    # service_times: pre-generated service times

    # output: 
    # n_blocked_customers : number of blocked customers

    # initialize variables
    n_blocked_customers = 0

    # initialize service units
    service_units = np.zeros(m)

    n_customers = len(interarrival_times)

    # iterate over customers
    time = 0
    for i in range(n_customers):
        # update time
        time += interarrival_times[i]

        # decrement the remaining service times of all occupied service units
        service_units = np.maximum(0, service_units - interarrival_times[i])

        # check if there are available service units
        if np.sum(service_units == 0) > 0:
            # assign service unit
            service_units[np.where(service_units == 0)[0][0]] = service_times[i]
        else:
            # block customer
            n_blocked_customers += 1

    return n_blocked_customers

# Control variate implementation
m = 10
mean_st = 8
mean_bc = 1
n_customers = 10 * 10000

# Set random seed for reproducibility
np.random.seed(42)

# Generate common random numbers for interarrival and service times
poisson_interarrival_times = np.random.exponential(mean_bc, n_customers)
uniform_random_numbers = np.random.uniform(size=n_customers)
lambda1 = 1 / 1.2
lambda2 = 1 / 5
hyperexponential_interarrival_times = np.where(
    uniform_random_numbers < 0.8,
    np.random.exponential(1 / lambda1, n_customers),
    np.random.exponential(1 / lambda2, n_customers)
)
service_times = np.random.exponential(mean_st, n_customers)

# Run simulations with Poisson arrival type
n_blocked_customers_poisson = simulation(m, mean_st, poisson_interarrival_times, service_times)
fraction_blocked_poisson = n_blocked_customers_poisson / n_customers

# Run simulations with hyperexponential arrival type
n_blocked_customers_hyperexponential = simulation(m, mean_st, hyperexponential_interarrival_times, service_times)
fraction_blocked_hyperexponential = n_blocked_customers_hyperexponential / n_customers

# Print results
print('Using common random numbers:')
print('Poisson arrival type - Fraction of blocked customers:', fraction_blocked_poisson)
print('Hyperexponential arrival type - Fraction of blocked customers:', fraction_blocked_hyperexponential)