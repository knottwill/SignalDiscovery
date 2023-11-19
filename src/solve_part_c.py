import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pytest import approx

def signal_pdf(M, mu, sigma, alpha, beta):
    """
    Signal Probability Density Function

    Computes the normalized probability density function of the
    signal, described by a normal distribution, within the range 
    [alpha, beta]

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    mu: float
        The mean of the normal distribution.
    sigma: float
        The standard deviation of the normal distribution.
    alpha: float
        The lower bound of the range of interest.
    beta: float
        The upper bound of the range of interest.

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

    Computes the normalized probability density function of the 
    background, described by an exponential decay distribution, 
    within the range [alpha, beta]

    Parameters
    -----------
    M: float or array-like
        The value(s) at which to evaluate the PDF.
    lam: float
        The decay constant of the exponential distribution.
    alpha: float
        The lower bound of the range of interest.
    beta: float
        The upper bound of the range of interest.

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
        The lower bound of the range of interest.
    beta: float
        The upper bound of the range of interest.

    Returns
    -----------
    float or array-like
        Combined PDF value(s) of the signal and background.
    """
    assert(f >= 0 and f <= 1)
    assert(lam > 0)
    assert(alpha > 0)
    assert(beta > 0)
    assert(beta > alpha)
    assert(sigma > 0)

    # Calculate the total PDF as a weighted sum of signal and background PDFs
    return f * signal_pdf(M, mu, sigma, alpha, beta) + (1 - f) * background_pdf(M, lam, alpha, beta)


# Defines upper and lower bounds
alpha = 5
beta = 5.6

# We perform the integration for 1000 random combinations of parameters
integrates_to_unity = True
N_combos = 1000
print(f'Integrating PDF over [{alpha}, {beta}] for {N_combos} random combinations of parameters...\n')
for _ in range(N_combos):

    # Generate parameter values from uniform distribution over an appropriate range
    f = np.random.uniform(0,1)
    lam = np.random.uniform(0,2)
    mu = np.random.uniform(alpha, beta)
    sigma = np.random.uniform(0.1,0.7)

    # Performs integration of total PDF over [5,5.6] given the generated parameters
    total_probability, error = quad(total_pdf, alpha, beta, args=(f, lam, mu, sigma, alpha, beta))

    # testing whether the total_probability is approximately 1
    # with a relative tolerance of 1e-6 (using pytest.approx)
    relative_tolerance = 1e-6
    if total_probability != approx(1, rel=relative_tolerance):

        print(f'Params: alpha = {alpha:.4}, beta = {beta:.4}, f = {f:.4}, lambda = {lam:.4}, mu = {mu:.4}, sigma = {sigma:.4}')
        print('Total Probability: {total_probability}')
        
        integrates_to_unity=False # failed

# Print statements if test was succeeded vs failed
if integrates_to_unity:
    print(f'As expected, the total PDF integrated to (approximately) unity on all')
    print(f'{N_combos} trials (with a relative tolerance of {relative_tolerance})\n')
else:
    print('Failed.')