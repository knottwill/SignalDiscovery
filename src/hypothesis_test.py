import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
from pytest import approx 
from distributions import total_pdf, total_cdf, two_signal_pdf, two_signal_cdf

def NLL(dataset, pdf, params):
    """
    Calculates the negative log likelihood (NLL), with a factor of 2

    This function calculates the NLL of a dataset given the probability
    density function and its parameters. A factor of 2 is added so that
    the difference between the NLL for two hypotheses is chi2 distributed.

    Parameters
    ----------
    dataset : np.ndarray
        The dataset for which the NLL is to be calculated.
    pdf : function
        The probability density function used to calculate the likelihood.
    params : list
        Parameters of the pdf

    Returns
    -------
    float
        The negative log likelihood, multiplied by factor 2
    """
    likelihood = pdf(dataset, *params)
    return -2*np.sum(np.log(likelihood))


def plotting(dataset, pdf, h0_params, h1_params):
    """
    Create plot for dataset with the H0 and H1 model overlaid
    """
    # square root rule for number of bins
    bins = int(np.sqrt(len(dataset)))

    # Bin the dataset
    bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)

    # Calculate bin midpoints
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots()

    # Plotting bin count vs midpoints
    x = np.linspace(5, 5.6, 1000)
    ax.scatter(midpoints, bin_density, label='Dataset', marker='o')
    # ax.plot(midpoints, total_model(midpoints, **true_params), label='True model', color='red')
    ax.plot(x, pdf(x, *h0_params), label='H0', color='red')
    ax.plot(x, pdf(x, *h1_params), label='H1', color='orange')

    ax.set_xlabel('M')
    ax.set_ylabel('Bin Density')
    ax.legend()

    return fig

def signal_background_test(dataset, starting_params, binned=False, plot=False):
    """
    Perform Neyman-Pearson Hypothesis test for the existence of a signal in a dataset.

    This function performs a hypothesis test to determine the presence of a signal 
    within the provided dataset. The null hypothesis is the background-only model and
    the alternate hypothesis is the signal + background model. The function fits these
    models to the dataset using maximum likelihood estimation via the iminuit package.
    The cost function is either the binned or unbinned negative log likelihood (specified by 'binned' Flag).
    The Neyman-Pearson test statistic is calculated as the difference in the negative log
    likelihoods between the two hypotheses, with a factor of 2. The p-value is calculated under 
    the assumption that this test statistic is chi2 distributed with 1 degree of freedom. 
    This is a one-sided test since the fraction of the signal can only be positive or zero, 
    hence the significance is calculated via the formula Z = norm.ppf(1 - p_value). If the 
    significance is greater than or equal to 5, then this consistutes a 'discovery'.

    Note: Even if a binned NLL cost function is used for minimisation, the test statistic
    is still calculated using the unbinned NLL. This was chosen since it provides more
    information, hence a more reliable test. 

    Parameters
    -------------
    dataset: np.ndarray
        The dataset on which to perform the hypothesis test
    starting_params: dict
        The starting parameters for the maximum likelihood estimation
    binned: bool
        Flag to indicate whether to use binned or unbinned negative log 
        likelihood as the cost function for minimisation
    plot: bool
        Flag to indicate whether the results of the fitting should be plotted

    Returns
    -------------
    tuple
        A tuple containing the results of the Neyman-Pearson hypothesis test 
        and (optionally) the plotted figure if plot=True. The elements of the tuple are:

        discovery : bool
            A boolean indicating whether the signal was 'discovered' 
            (True if the significance Z is greater than or equal to 5, False otherwise).

        Z : float
            The calculated significance of the test, representing the number of standard deviations 
            the test statistic is away from the mean of the null distribution.

        p_value : float
            The p-value of the test, representing the probability of observing a test statistic 
            as extreme as, or more extreme than, the observed value under the null hypothesis.

        fig : matplotlib.figure.Figure, optional
            A figure object showing the fitted results. Only returned if 'plot=True'
    """
    # --------------------
    # Minimisation object
    # --------------------

    # Cost function is either unbinned or binned negative log likelihood
    if binned:
        bins = int(np.sqrt(len(dataset))) # Square root rule for number of bins
        bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)
        nll = BinnedNLL(bin_density, bin_edges, total_cdf) # Binned negative log likelihood
    else:
        nll = UnbinnedNLL(dataset, total_pdf)
    
    # Minimisation object
    mi = Minuit(nll, **starting_params)
        
    # Setting constraints
    alpha = 5
    beta = 5.6
    mi.limits['f'] = (0.01, 1) # fraction of signal is between 0 and 1
    mi.limits['lam'] = (0.1, 1.5) # lambda cannot be negative (otherwise there is no 'decay')
    mi.limits['sigma'] = (0.01, (beta-alpha)/2) # sigma should not be too wide, and cannot be negative
    mi.limits['mu'] = (alpha, beta) # the signal should not peak outside of [alpha, beta]

    # ---------------------------
    # Fit the alternate hypothesis (signal + background)
    # ---------------------------

    # Run minimisation and Hesse algorithm
    mi.migrad()
    mi.hesse()

    # Ensure that valid minimum has been found
    if not(mi.valid):
        return 'invalid minimum', 0, 0 # don't continue with test

    # Parameter values for total model fit
    h1_params = list(mi.values)

    # ---------------------------
    # Fit the null hypothesis (background-only)
    # ---------------------------

    # Set parameters such that we fit background-only model
    mi.values['f'] = 0
    mi.fixed['f'] = True
    mi.fixed['mu'] = True
    mi.fixed['sigma'] = True

    # Run minimisation and Hesse algorithm
    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    if not(mi.valid):
        return 'invalid minimum', 0, 0 # don't continue with test

    # Parameter values for background-only fit
    h0_params = list(mi.values)

    # ---------------------------
    # Perform Neyman-Pearson Test
    # ---------------------------

    # Calculate negative log likelihoood (with factor 2) for each hypothesis 
    h0_nll = NLL(dataset, total_pdf, h0_params)
    h1_nll = NLL(dataset, total_pdf, h1_params)

    # Neyman-Pearson Test statistic
    T = h0_nll - h1_nll

    # Calculate p value
    k = 1 # degrees of freedom
    p_value = 1 - chi2.cdf(T, k) 

    # Calculate significance (one-sided test)
    Z = norm.ppf(1 - p_value)

    # If we get a significance greater than 5, we have 'discovered' the signal
    if Z >= 5:
        discovery = True
    else:
        discovery = False

    if plot:
        fig = plotting(dataset, total_pdf, h0_params, h1_params)
        return fig, discovery, Z, p_value

    return (discovery, Z, p_value)


def two_signal_test(dataset, starting_params, binned=False, plot=False):
    """
    Perform Neyman-Pearson Hypothesis test for the existance of two distinct signals

    This function performs a hypothesis test to determine the presence of two distinct
    signals within the provided dataset. The null hypothesis is the 'background + one
    signal' model and the alternate hypothesis is the 'background + two signals' model.
    The function fits these models to the dataset using maximum likelihood estimation
    via the iminuit package. The cost function is either a binned or unbinned negative
    log likelihood (specified by 'binned' Flag). 
    The Neyman-Pearson test statistic is calculated as the difference in the negative log
    likelihoods between the two hypotheses, with a factor of 2. The p-value is calculated under 
    the assumption that this test statistic is chi2 distributed with 1 degree of freedom. 
    This is a one-sided test since the fraction of the signal can only be positive or zero, 
    hence the significance is calculated via the formula Z = norm.ppf(1 - p_value). If the 
    significance is greater than or equal to 5, then this consistutes a 'discovery'.

    Parameters
    -------------
    dataset: np.ndarray
        The dataset on which to perform the hypothesis test
    starting_params: dict
        The starting parameters for the maximum likelihood estimation
    binned: bool
        Flag to indicate whether to use binned or unbinned negative log 
        likelihood as the cost function for minimisation
    plot: bool
        Flag to indicate whether the results of the fitting should be plotted

    Returns
    -------------
    tuple
        A tuple containing the results of the Neyman-Pearson hypothesis test 
        and (optionally) the plotted figure if plot=True. The elements of the tuple are:

        discovery : bool
            A boolean indicating whether the signal was 'discovered' 
            (True if the significance Z is greater than or equal to 5, False otherwise).

        Z : float
            The calculated significance of the test, representing the number of standard deviations 
            the test statistic is away from the mean of the null distribution.

        p_value : float
            The p-value of the test, representing the probability of observing a test statistic 
            as extreme as, or more extreme than, the observed value under the null hypothesis.

        fig : matplotlib.figure.Figure, optional
            A figure object showing the fitted results. Only returned if 'plot=True'
    """

    # --------------------
    # Minimisation object
    # --------------------

    # Cost function is either unbinned or binned negative log likelihood
    if binned:
        bins = int(np.sqrt(len(dataset))) # Square root rule for number of bins
        bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)
        nll = BinnedNLL(bin_density, bin_edges, two_signal_cdf) # Binned negative log likelihood
    else:
        nll = UnbinnedNLL(dataset, two_signal_pdf)
    
    # Minimisation object
    mi = Minuit(nll, **starting_params)

    # Setting constraints
    mi.limits['f1'] = (0.01, 0.5)
    mi.limits['f2'] = (0.01, 0.5)
    mi.limits['mu1'] = (5, 5.6) # the signal should not peak outside of [alpha, beta]
    mi.limits['mu2'] = (5, 5.6)
    mi.limits['lam'] = (0.1, 1.5) # lambda cannot be negative (otherwise there is no 'decay')
    mi.limits['sigma'] = (0.01, 0.3) # sigma should not be too wide, and cannot be negative

    # ---------------------------
    # Run the fit for the alternate hypothesis
    # ---------------------------

    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    if not(mi.valid):
        return 'invalid minimum', 0, 0 # don't continue with test

    # Estimated parameters for H1 model
    h1_params = list(mi.values)

    # ---------------------------
    # Run the fit for the null hypothesis
    # ---------------------------
    mi.values['f2'] = 0
    mi.fixed['f2'] = True
    mi.fixed['mu2'] = True

    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    if not(mi.valid):
        return 'invalid minimum', 0, 0 # don't continue with test

    # Estimated parameters for H0 model
    h0_params = list(mi.values)

    # ---------------------------
    # Perform Neyman-Pearson Test
    # ---------------------------

    # Calculate negative log likelihoood (with factor 2) for each hypothesis 
    h0_nll = NLL(dataset, two_signal_pdf, h0_params)
    h1_nll = NLL(dataset, two_signal_pdf, h1_params)

    # Neyman-Pearson Test statistic
    T = h0_nll - h1_nll

    # Calculate p value
    k = 1 # degrees of freedom
    p_value = 1 - chi2.cdf(T, k) 

    # Calculate significance (one-sided test)
    Z = norm.ppf(1 - p_value)

    # If we get a significance greater than 5, we have 'discovered' the signal
    if Z >= 5:
        discovery = True
    else:
        discovery = False

    if plot:
        fig = plotting(dataset, two_signal_pdf, h0_params, h1_params)
        return fig, discovery, Z, p_value

    return (discovery, Z, p_value)
