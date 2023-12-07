"""
This module is used to generate datasets using the PDFs relevant 
to the coursework problem
"""

import numpy as np
from scipy.stats import norm, expon
from time import process_time

def generate_from_total_pdf(N_events, f=0.1, lam=0.5, mu=5.28, sigma=0.018):
    """
    Inverse CDF method to generate data from the total (signal + background) PDF

    Since we do not have the percentage point function (inverse CDF) of the total
    signal + background model, we must use the PPFs of the individual signal and
    background models. This generation algorithm works by randomly choosing to 
    generate each data point from the signal-only model or background-only model
    with probability 0.1 and 0.9. Since the CDFs and PPFs of the scipy normal and 
    exponential distributions are not automatically normalised over [5, 5.6], we 
    generate from each distribution by first evaluating the non-normalised CDFs at
    alpha = 5 and beta = 5.6, then generating a probability uniformly in this range.
    This probability is then fed into the non-normalised inverse CDFs to generate an
    event. This works because the weightings for the total model are such that the signal 
    contributes 0.1 to the total probability, and the background contributes 0.9. 

    For a more detailed explanation, please refer to part e of the report. 
    """

    # upper & lower bounds of M
    alpha = 5
    beta = 5.6

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

    # Generating signal events using percentage point function
    probs = np.random.uniform(lower_p_signal, upper_p_signal, N_signal_events)
    signal_events = norm.ppf(q=probs, loc=mu, scale=sigma)

    # Generating background events using percentage point function
    probs = np.random.uniform(lower_p_background, upper_p_background, N_background_events)
    background_events = expon.ppf(q=probs, scale=1/lam)

    # Total events is just the union of signal and background events
    total_events = np.concatenate((signal_events, background_events))

    return total_events


def generate_from_two_signal_pdf(N_events, f1=0.1, f2=0.05, lam=0.5, mu1=5.28, mu2=5.35, sigma=0.018):
    """
    Inverse CDF method to generate data from the combined PDF of two signals and background

    The algorithm is the same as 'generate_from_total_pdf'
    """

    # upper & lower bounds of M
    alpha = 5
    beta = 5.6

    # Randomly select the number of signal events and background events
    # to generate, according to the weighting f
    g = np.random.uniform(0,1, N_events)
    N_s1 = np.count_nonzero(g <= f1)
    N_s2 = np.count_nonzero((f1 < g) & (g <= f1 + f2))
    N_background = np.count_nonzero(g > f1+f2)

    # Finding lower and upper bounds of the probabilities
    lower_p_s1 = norm.cdf(alpha, loc=mu1, scale=sigma)
    upper_p_s1 = norm.cdf(beta, loc=mu1, scale=sigma)
    lower_p_s2 = norm.cdf(alpha, loc=mu2, scale=sigma)
    upper_p_s2 = norm.cdf(beta, loc=mu2, scale=sigma)
    lower_p_background = expon.cdf(alpha, scale=1/lam)
    upper_p_background = expon.cdf(beta, scale=1/lam)

    # Generating s1 events using percentage point function
    probs = np.random.uniform(lower_p_s1, upper_p_s1, N_s1)
    s1_events = norm.ppf(q=probs, loc=mu1, scale=sigma)

    # Generating s2 events using percentage point function
    probs = np.random.uniform(lower_p_s2, upper_p_s2, N_s2)
    s2_events = norm.ppf(q=probs, loc=mu2, scale=sigma)

    # Generating background events using percentage point function
    probs = np.random.uniform(lower_p_background, upper_p_background, N_background)
    background_events = expon.ppf(q=probs, scale=1/lam)

    # Total events is just the union of all events
    total_events = np.concatenate((s1_events, s2_events, background_events))

    return total_events