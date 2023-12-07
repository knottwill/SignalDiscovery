"""
Distributions (PDFs and CDFs)
----------
This module contains functions that compute the normalised probability
density functions and cumulative density functions of the background-only 
model, the signal model, the signal + background model, and the double signal
model. All are defined over the range [5, 5.6]. This range must not be a parameter,
otherwise they cannot be used to fit to data.
"""

from scipy.stats import norm, expon

def signal_pdf(M, mu, sigma):
    """
    Signal Probability Density Function

    Computes the probability density function of the signal, 
    described by a normal distribution, normalised within the 
    range [5, 5.6].

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    mu: float
        The mean of the normal distribution.
    sigma: float
        The standard deviation of the normal distribution.

    Returns
    ----------
    float or array-like
        Normalized PDF value(s) of the signal.
    """
    # upper and lower bound of M
    alpha = 5
    beta = 5.6

    # Calculate the total probability within [alpha, beta] for the normal distribution
    total_prob = norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma)

    # Normalization factor to ensure the PDF integrates to 1 within [alpha, beta]
    normalisation_factor = 1 / total_prob

    # Return the normalized PDF values for the given values of M
    return normalisation_factor * norm.pdf(x=M, loc=mu, scale=sigma)
    

def background_pdf(M, lam):
    """
    Background Probability Density Function

    Computes the probability density function of the background, 
    described by an exponential decay distribution, normalised
    within the range [5, 5.6]

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    lam: float
        The decay constant of the exponential distribution.

    Returns
    -----------
    float or array-like
        Normalized PDF value(s) of the background.
    """
    # upper and lower bound of M
    alpha = 5
    beta = 5.6

    # Calculate the total probability within [alpha, beta] for the exponential distribution
    total_prob = expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam)

    # Normalization factor to ensure the PDF integrates to 1 within [alpha, beta]
    normalisation_factor = 1 / total_prob

    # Return the normalized PDF values for the given values of M
    return normalisation_factor * expon.pdf(x=M, scale=1/lam)


def total_pdf(M, f, lam, mu, sigma):
    """
    Total (signal + background) Probability Density Function

    Computes the signal + background PDF as a weighted sum of the normalised
    signal and background PDFs

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    f: float
        Fraction of the signal in the total distribution.
    lam: float
        The decay constant of the exponential distribution for the background.
    mu: float
        The mean of the normal distribution for the signal.
    sigma: float
        The standard deviation of the normal distribution for the signal.

    Returns
    -----------
    float or array-like
        Combined PDF value(s) of the signal and background.
    """

    # The total PDF is a weighted sum of signal and background PDFs
    return f * signal_pdf(M, mu, sigma) + (1 - f) * background_pdf(M, lam)

def total_cdf(M, f, lam, mu, sigma):
    """
    Total (signal + background) cumulative density function

    Computes the CDF of the signal + background model as a weighted sum of
    the CDFs of the normalised signal-only and background-only model

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    f: float
        Fraction of the signal in the total distribution.
    lam: float
        The decay constant of the exponential distribution for the background.
    mu: float
        The mean of the normal distribution for the signal.
    sigma: float
        The standard deviation of the normal distribution for the signal.

    Returns
    -----------
    float or array-like
        CDF value(s) of the signal + background model
    """
    # Upper & lower bound of M
    alpha = 5
    beta = 5.6

    # normalisation factors for the signal and background distributions
    signal_factor = 1/(norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma))
    background_factor = 1/(expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    # CDFs of the signal-only and background-only models
    signal_cdf = signal_factor*(norm.cdf(x=M, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma))
    background_cdf = background_factor*(expon.cdf(x=M, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    # Total CDF is a weighted sum of the individual CDFs
    return f*signal_cdf + (1-f)*background_cdf

def two_signal_pdf(M, f1=0.1, f2=0.05, lam=0.5, mu1=5.28, mu2=5.35, sigma=0.018):
    """
    Probability Density Function of the double signal + background model

    Computes the PDF of the model consisting of background and two distinct signals.

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    f1: float
        Fraction of the first signal.
    f2: float
        Fraction of the second signal.
    lam: float
        The decay constant of the exponential distribution for the background.
    mu1: float
        The mean of the normal distribution for the first signal.
    mu2: float
        The mean of the normal distribution for the second signal.
    sigma: float
        The standard deviation of the normal distribution for both signals.

    Returns
    -----------
    float or array-like
        Combined PDF value(s) of the two signals and background.
    """

    # Total PDF is a weighted sum of the individually normalised PDFs
    return f1*signal_pdf(M, mu1, sigma) + f2*signal_pdf(M, mu2, sigma) + (1-f1-f2)*background_pdf(M, lam)

def two_signal_cdf(M, f1, f2, lam, mu1, mu2, sigma):
    """
    Cumulative density function of the double signal + background model

    Computes the CDF of the two signal + background model as a weighted sum of
    the CDFs of the individually normalised signal-only models and background-only 
    model

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    f1: float
        Fraction of the first signal.
    f2: float
        Fraction of the second signal.
    lam: float
        The decay constant of the exponential distribution for the background.
    mu1: float
        The mean of the normal distribution for the first signal.
    mu2: float
        The mean of the normal distribution for the second signal.
    sigma: float
        The standard deviation of the normal distribution for both signals.

    Returns
    -----------
    float or array-like
        CDF value(s) of the two signals + background model
    """
    # Upper & lower bound of M
    alpha = 5
    beta = 5.6

    # Normalisation factors for the signal and background distributions
    s1_factor = 1/(norm.cdf(x=beta, loc=mu1, scale=sigma) - norm.cdf(x=alpha, loc=mu1, scale=sigma))
    s2_factor = 1/(norm.cdf(x=beta, loc=mu2, scale=sigma) - norm.cdf(x=alpha, loc=mu2, scale=sigma))
    background_factor = 1/(expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    # CDFs of the two signal-only models and the background-only model
    s1_cdf = s1_factor*(norm.cdf(x=M, loc=mu1, scale=sigma) - norm.cdf(x=alpha, loc=mu1, scale=sigma))
    s2_cdf = s2_factor*(norm.cdf(x=M, loc=mu2, scale=sigma) - norm.cdf(x=alpha, loc=mu2, scale=sigma))
    background_cdf = background_factor*(expon.cdf(x=M, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    # Total CDF is a weighted sum of the individual CDFs
    return f1*s1_cdf + f2*s2_cdf + (1-f1-f2)*background_cdf