"""
This module is used to generate datasets using the PDFs relevant 
to the coursework problem
"""

import numpy as np
from scipy.stats import norm, expon
from time import process_time

def generate_from_total_pdf(N_events, f=0.1, lam=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, print_timings=False):
    """
    Inverse CDF method to generate data from the total PDF

    Finish docstring. talk about:
    - randomly selecting between signal or background to generate from
    - why this is the same as generating N_events total PDF events
    - How the inverse CDF is called the percentage point function, ppf

    ########################
    # Lower and upper bounds for the probability to feed
    # into the inverse CDF function can be found by
    # evaulating the CDFs at alpha and beta
    ########################
    """

    # Randomly select the number of signal events and background events
    # to generate, according to the weighting f
    g = np.random.uniform(0,1, N_events)
    N_background_events = np.count_nonzero(g > f)
    N_signal_events = np.count_nonzero(g <= f)

    # Finding lower and upper bounds of the probabilities to input into 
    # the norm and expon ppfs
    lower_p_signal = norm.cdf(alpha, loc=mu, scale=sigma)
    upper_p_signal = norm.cdf(beta, loc=mu, scale=sigma)
    lower_p_background = expon.cdf(alpha, scale=1/lam)
    upper_p_background = expon.cdf(beta, scale=1/lam)

    # Measure time to generate data
    start = process_time()

    # Generating signal events using percentage point function
    probs = np.random.uniform(lower_p_signal, upper_p_signal, N_signal_events)
    signal_events = norm.ppf(q=probs, loc=mu, scale=sigma)

    # Generating background events using percentage point function
    probs = np.random.uniform(lower_p_background, upper_p_background, N_background_events)
    background_events = expon.ppf(q=probs, scale=1/lam)

    # Total events is just the union of signal and background events
    total_events = np.concatenate((signal_events, background_events))

    # Print time
    stop = process_time()
    if print_timings:
        print(f'Generated {N_events} events in {stop-start:.4}s')

    return total_events