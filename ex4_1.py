import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulation(m, mean_st, mean_bc, n_customers, a_type, s_type):

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

    return n_blocked_customers


# Part 1
# parameters: 
# m = 10
# mean_st = 8
# mean_bc = 1
# n_customers = 10*10.000
# a_type = 'poisson'
# s_type = 'exponential'

# report the number of blocked customers, and a confidence interval for this fraction
m = 10
mean_st = 8
mean_bc = 1
n_customers = 10*10000
a_type = 'poisson'
s_type = 'exponential'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)


print('poisson arrival time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# compute analytical solution for the fraction of blocked customers using Erlang's B formula
A = 1 / mean_bc * mean_st

B = A ** m / np.math.factorial(m) / np.sum([A ** i / np.math.factorial(i) for i in range(m + 1)])
P_0 = 1 / (B + np.sum([A ** i / np.math.factorial(i) for i in range(m)]))
P_m = P_0 * A ** m / np.math.factorial(m)
fraction_blocked_analytical = P_m

print('Analytical fraction of blocked customers: ', fraction_blocked_analytical)

# part 2 a)

# report the number of blocked customers, and a confidence interval for this fraction
a_type = 'erlang'
s_type = 'exponential'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('erlang arrival time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# part 2 b)

# report the number of blocked customers, and a confidence interval for this fraction
a_type = 'hyperexponential'
s_type = 'exponential'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('hyperexponential arrival time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# part 3 a)
a_type = 'poisson'
s_type = 'constant'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('Constant service time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# part 3 b)
a_type = 'poisson'
s_type = 'pareto1'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)


print('pareto1 service time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# part 3 c)
a_type = 'poisson'
s_type = 'pareto2'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('pareto2 service time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)

# part 3 d)
a_type = 'poisson'
s_type = 'weibull'

# run simulation
n_blocked_customers = simulation(m, mean_st, mean_bc, n_customers, a_type, s_type)

# report the fraction of blocked customers and a 95% confidence interval for this fraction
fraction_blocked = n_blocked_customers / n_customers
std_error = np.sqrt(fraction_blocked * (1 - fraction_blocked) / n_customers)
confidence_interval = (fraction_blocked - 1.96 * std_error, fraction_blocked + 1.96 * std_error)

print('weibull service time')
print('Fraction of blocked customers: ', fraction_blocked)
print('Confidence interval: ', confidence_interval)
