import numpy as np
import matplotlib.pyplot as plt

# Standard binomial error
# on estimate of 'probability of discovery'
def standard_binomial_error(p, n):
    return np.sqrt((p*(1-p))/n)

def probability_of_discovery(N_events, n_attempts, true_params, generation_func, hypothesis_test, pdf, cdf):
    """
    Find probability of discovering the signal for each data size 'N_events'

    This function estimates the probability of discovering the signal by
    performing a Neyman-Pearson test on a dataset of size 'N_events'
    """
    # ------------------------------
    # Generating starting parameters for minimisation
    # by adding random shifts of up to 30% of true params
    # ------------------------------

    starting_params = {}
    for param in true_params:
        max_shift = 0.3*true_params[param]
        random_shift = np.random.uniform(-max_shift, max_shift)
        starting_params[param] = true_params[param] + random_shift
        starting_params[param] = float(f'{starting_params[param]:.4}') # rounding to 4 sig. fig.

    # ------------------------------
    # Generation and Hypothesis Testing
    # ------------------------------
    discovery_count = 0 # number of 'discoveries'
    for _ in range(n_attempts):

        dataset = generation_func(N_events)

        discovery, _, _ = hypothesis_test(
            dataset=dataset, 
            pdf=pdf, 
            cdf=cdf, 
            starting_params=true_params,
            binned=False, 
            plot=False
            )
        
        # if a valid minimum was not found 
        # then it is not a valid hypothesis test 
        if discovery=='invalid minimum':
            n_attempts -= 1

        if discovery==True:
            discovery_count += 1

    p = discovery_count/n_attempts
    p_err = standard_binomial_error(p, n_attempts)

    print(f"number of successful hypothesis tests conducted: {n_attempts}")

    return p, p_err