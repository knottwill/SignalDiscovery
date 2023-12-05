import numpy as np
import matplotlib.pyplot as plt
from generation import generate_from_total_pdf
from distributions import total_cdf, total_pdf
from hypothesis_test import signal_background_test

def generate_NP_data(n_datapoints, n_attempts, N_range):
    """
    Generate N vs P data

    This function estimates the probability of discovering the signal
    for a range of dataset sizes, by performing Neyman-Pearson 
    hypothesis tests.
    """
    # True parameter values
    true_params, alpha, beta = {'f': 0.1, 'lam': 0.5, 'mu': 5.28, 'sigma': 0.018}, 5, 5.6

    # Generating starting parameters by adding random shifts to true parameters
    np.random.seed(42)
    random_shifts = {
        'f': np.random.uniform(-0.09, 0.5), 
        'lam': np.random.uniform(-0.3, 0.6), 
        'mu': np.random.uniform(-0.25, 0.3), 
        'sigma': np.random.uniform(-0.01, 0.03)
    }
    starting_params = {}
    for param in true_params:
        starting_params[param] = true_params[param] + random_shifts[param]
        starting_params[param] = float(f'{starting_params[param]:.4}') # rounding to 4 sig. fig.

    # Standard binomial error
    # on estimate of 'probability of discovery'
    def standard_binomial_error(p, n):
        return np.sqrt((p*(1-p))/n)

    # -----------------------------
    # Data Generation and Hypothesis Testing
    # -----------------------------
    N = np.linspace(N_range[0], N_range[1], n_datapoints).astype(int) # values of 'N_events'
    P = [] # probability of discovery
    P_err = [] # error in probability of discovery
    for N_events in N:
        n_i = n_attempts # number of successful hypothesis tests conducted
        discovery_count = 0 # number of 'discoveries'
        for _ in range(n_attempts):

            dataset = generate_from_total_pdf(N_events)

            discovery, _, _ = signal_background_test(
                dataset=dataset, 
                pdf=total_pdf, 
                cdf=total_cdf, 
                starting_params=true_params,
                binned=False, 
                plot=False
                )
            
            # if a valid minimum was not found 
            # then it is not a valid hypothesis test 
            if discovery=='invalid minimum':
                n_i -= 1

            if discovery==True:
                discovery_count += 1

        p = discovery_count/n_i
        P.append(p)
        P_err.append(standard_binomial_error(p, n_i))
        print(f"number of attempts: {n_i}")
        print(f"Sample size (N_events)={N_events}, probability of discovery={P[-1]} +- {P_err[-1]}")

    return np.array(N), np.array(P), np.array(P_err)