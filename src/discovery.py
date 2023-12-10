import numpy as np
import matplotlib.pyplot as plt

def probability_of_discovery(N_events, n_trials, true_params, generation_func, hypothesis_test):
    """
    Estimate Probability of 'discovering' the alternate hypothesis for a dataset of size N_events

    This function repeats the following procedure for 'n_trials' iterations:
    - generate dataset of size 'N_events' using 'generation_func'
    - Perform hypothesis test on this dataset using 'hypothesis_test'. 
    - If the alternate hypothesis was accepted with a significance of Z > 5, then it is a 'discovery'
    The function counts the number of discoveries out of 'n_trials' and estimates the probability
    of discovery as:
    p = (number of discoveries)/(n_trials)
    Since the number of discoveries is binomially distributed, the uncertainty of p is calculated
    using the standard deviation of the binomial distribution:
    p_err = sqrt(p*(1-p) / n_trials)

    Parameters
    -------------
    N_events: int
        The size of the datasets to generate (number of events)
    n_trials: int
        The number of times to generate the dataset and perform the hypothesis test on it
    true_params: dict
        The true parameters of the PDF
    generation_func: function
        The function to generate the datasets
    hypothesis_test: function
        The function to perform the hypothesis tests on the data

    Returns
    ---------
    tuple containing elements:
        - probability of discovery: p
        - uncertainty of probability: p_err
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

    # calculate probability and uncertainty
    p = discovery_count/n_trials
    p_err = np.sqrt((p*(1-p))/n_trials)

    return (p, p_err)