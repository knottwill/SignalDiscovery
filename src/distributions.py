"""
PDFs
----------
This module contains functions that compute the normalised probability
density functions, within specified ranges, of the signal-only PDF, 
background-only PDF and total PDF (which is a weighted sum of signal 
and background).
"""

from scipy.stats import norm, expon

def signal_pdf(M, mu, sigma, alpha, beta):
    """
    Signal Probability Density Function

    Computes the probability density function of the signal, 
    described by a normal distribution, normalised within the 
    range [alpha, beta].

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    mu: float
        The mean of the normal distribution.
    sigma: float
        The standard deviation of the normal distribution.
    alpha: float
        The lower bound of M.
    beta: float
        The upper bound of M.

    Returns
    ----------
    float or array-like
        Normalized PDF value(s) of the signal.
    """
    assert(alpha > 0)
    assert(beta > 0)
    assert(beta > alpha)
    assert(sigma > 0)

    # Calculate the total probability within [alpha, beta] for the normal distribution
    total_prob = norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma)

    # Normalization factor to ensure the PDF integrates to 1 within [alpha, beta]
    normalisation_factor = 1 / total_prob

    # Return the normalized PDF values for the given values of M
    return normalisation_factor * norm.pdf(x=M, loc=mu, scale=sigma)
    

def background_pdf(M, lam, alpha, beta):
    """
    Background Probability Density Function

    Computes the probability density function of the background, 
    described by an exponential decay distribution, normalised
    within the range [alpha, beta]

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    lam: float
        The decay constant of the exponential distribution.
    alpha: float
        The lower bound of M.
    beta: float
        The upper bound of M.

    Returns
    -----------
    float or array-like
        Normalized PDF value(s) of the background.
    """

    assert(lam > 0)
    assert(alpha > 0)
    assert(beta > 0)
    assert(beta > alpha)

    # Calculate the total probability within [alpha, beta] for the exponential distribution
    total_prob = expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam)

    # Normalization factor to ensure the PDF integrates to 1 within [alpha, beta]
    normalisation_factor = 1 / total_prob

    # Return the normalized PDF values for the given values of M
    return normalisation_factor * expon.pdf(x=M, scale=1/lam)


def total_pdf(M, f, lam, mu, sigma, alpha, beta):
    """
    Computes the combined probability density function (PDF) for a mixture
    of a signal and background within a specified range.

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
    alpha: float
        The lower bound of M.
    beta: float
        The upper bound of M.

    Returns
    -----------
    float or array-like
        Combined PDF value(s) of the signal and background.
    """
    assert(f >= 0 and f <= 1)

    # Calculate the total PDF as a weighted sum of signal and background PDFs
    return f * signal_pdf(M, mu, sigma, alpha, beta) + (1 - f) * background_pdf(M, lam, alpha, beta)


def shifted_background_pdf(M, lam, alpha, beta):

    assert(lam > 0)
    assert(alpha > 0)
    assert(beta > 0)
    assert(beta > alpha)

    # same calculation as above, but expon distribution is shifted to begin at alpha
    total_prob = expon.cdf(x=beta, loc=alpha, scale=1/lam) - expon.cdf(x=alpha, loc=alpha, scale=1/lam)
    normalisation_factor = 1 / total_prob
    return normalisation_factor * expon.pdf(x=M, loc=alpha, scale=1/lam)

def signal_model(M, mu, sigma):
    """
    For minimisation using iminuit, we can't have alpha and beta be parameters
    We also can't have assert statements
    """
    alpha = 5
    beta = 5.6

    total_prob = norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma)
    normalisation_factor = 1 / total_prob
    return normalisation_factor * norm.pdf(x=M, loc=mu, scale=sigma)
    

def background_model(M, lam):
    """
    """
    alpha = 5
    beta = 5.6

    total_prob = expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam)
    normalisation_factor = 1 / total_prob
    return normalisation_factor * expon.pdf(x=M, scale=1/lam)


def total_model(M, f, lam, mu, sigma):
    """
    """

    return f * signal_model(M, mu, sigma) + (1 - f) * background_model(M, lam)

def total_cdf(M, f, lam, mu, sigma):
    alpha = 5
    beta = 5.6

    # normalisation factors for the signal and background distributions
    signal_factor = 1/(norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma))
    background_factor = 1/(expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    signal_cdf = signal_factor*(norm.cdf(x=M, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma))
    background_cdf = background_factor*(expon.cdf(x=M, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam))

    return f*signal_cdf + (1-f)*background_cdf