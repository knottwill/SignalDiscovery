import numpy as np
import matplotlib.pyplot as plt

# Standard binomial error
# on estimate of 'probability of discovery'
def standard_binomial_error(p, n):
    return np.sqrt((p*(1-p))/n)

def probability_of_discovery(N_events, n_trials, true_params, generation_func, hypothesis_test):
    """
    Estimate Probability of 'discovering' the alternate hypothesis for a dataset of size N_events

    This function generates a dataset of size 'N_events' using the generation function
    provided 'generation_func'. It then performs a hypothesis test on this dataset using the function
    'hypothesis_test' to see if the alternate hypothesis was accepted with a 
    significance of Z > 5 (a 'discovery'). It then repeats this process for 'n_trials'
    iterations, counts the number of times a discovery was made, and calculates the probability
    of making a discovery for the given dataset size with the equation:
    p = (number of discoveries)/(number of attempts)
    Since this is a series of bernoulli trials, we can estimate the uncertainty on this 
    probability estimate as the standard binomial error:
    p_err = sqrt(p*(1-p) / n_trials)

    Parameters
    -------------
    N_events: int
        The size of the datasets (number of events)
    n_trials: int
        The number of times to generate the dataset and perform the hypothesis test on it
    true_params: dict
        The true parameters of the PDF
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
    for _ in range(n_trials):

        dataset = generation_func(N_events)

        discovery, _, _ = hypothesis_test(
            dataset=dataset, 
            starting_params=true_params,
            binned=False, 
            plot=False
            )
        
        # if a valid minimum was not found 
        # then it is not a valid hypothesis test 
        if discovery=='invalid minimum':
            n_trials -= 1

        if discovery==True:
            discovery_count += 1

    p = discovery_count/n_trials
    p_err = standard_binomial_error(p, n_trials)

    return p, p_err