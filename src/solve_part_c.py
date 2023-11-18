import numpy as np
from scipy.stats import norm, expon
from scipy.integrate import quad

def normalisation_factor(f, lam, mu, sigma, alpha, beta):
    """
    Normalisation Factor

    Calculates the normalisation factor for the total probability distribution 
    (ie. signal + background distribution), given the upper and lower bounds: alpha, beta.
    We are assuming that alpha and beta are positive

    Parameters
    -----------
    f: float
        the fraction of signal, must be within [0, 1]
    lam
        decay constant (lambda), must be positive
    mu
        mean for normal distribution
    sigma
        standard deviation for normal distribution
    alpha
        lower bound of random variable
    beta
        upper bound of random variable

    Returns
    -----------
    float
        the normalisation factor
    """
    assert(f >= 0 and f <= 1)
    assert(lam > 0)
    assert(alpha > 0)
    assert(beta > 0)
    assert(sigma > 0)

    # calculates probability of "x in [alpha, beta]" for normal distribution
    norm_prob = norm.cdf(x=beta, loc=mu, scale=sigma) - norm.cdf(x=alpha, loc=mu, scale=sigma)

    # calculates probability of "x in [alpha, beta]" for exponential decay distribution
    expon_prob = expon.cdf(x=beta, scale=1/lam) - expon.cdf(x=alpha, scale=1/lam)

    return 1/(f*norm_prob + (1-f)*expon_prob)

##############################
### add comments/docstrings here
###############################
def total_pdf(M, f, lam, mu, sigma, alpha, beta):

    A = normalisation_factor(f, lam, mu, sigma, alpha, beta)

    return A*(f*norm.pdf(x=M, loc=mu, scale=sigma) + (1-f)*expon.pdf(x=beta, scale=1/lam))