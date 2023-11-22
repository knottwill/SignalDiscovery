import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
from pytest import approx 

def neyman_pearson_test(h0_nll, h1_nll):
    # ---------------------------
    # Perform Neyman-Pearson Test
    # ---------------------------

    T = h0_nll - h1_nll # test statistic
    k = 1               # degrees of freedom

    # Calculate p value
    p_value = 1 - chi2.cdf(T, k) 

    # Calculate significance, only if the p value is larger than 0.5 
    # (otherwise the significance is meaningless)
    # (it is a one-sided test)
    Z = norm.ppf(1 - p_value)

    # Alternate way to calculate Z for a one-sided test
    # uses the chi2 percentage point function
    # we assert that the significances are the same 
    # (except if the p value is greater than 0.5, in which case 1 - 2*p_value evaluates to 
    # a negative probability, which doesn't make sense)
    if p_value < 0.5:
        alternate_Z = np.sqrt(chi2.ppf(1 - 2*p_value,1))
        assert(Z == approx(alternate_Z))

    return Z, p_value

def plotting(dataset, pdf, h0_params, h1_params):
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

def signal_background_test(dataset, pdf, cdf, starting_params, binned=False, plot=False):
    """
    Perform Neyman-Pearson Hypothesis test for the existance of a signal

    Points to hit on in this docstring:
    - We construct H0 and H1 by doing binned maximum likelihood estimation
    - test statistic is chi2 distributed with 1 dof
    - its a one-sided test, hence Z = np.sqrt(chi2.ppf(1 - 2*p_value,1))
    """
    # upper and lower bounds
    alpha = 5
    beta = 5.6

     # --------------------
    # Minimisation object
    # --------------------

    if binned:
            # square root rule for number of bins
        bins = int(np.sqrt(len(dataset)))
        
        # Bin the dataset
        bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)

        # Calculate bin midpoints
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Cost function is binned negative log likelihood
        nll = BinnedNLL(bin_density, bin_edges, cdf)

        # Minimisation object
        mi = Minuit(nll, **starting_params)

    else:
        nll = UnbinnedNLL(dataset, pdf)
        mi = Minuit(nll, **starting_params)
        

    # Setting constraints
    mi.limits['f'] = (0, 1) # fraction of signal is between 0 and 1
    mi.limits['lam'] = (0, None) # lambda cannot be negative (otherwise there is no 'decay')
    mi.limits['sigma'] = (0, (beta-alpha)/2) # sigma should not be too wide, and cannot be negative
    mi.limits['mu'] = (alpha, beta) # the signal should not peak outside of [alpha, beta]

    # ---------------------------
    # Run the fit for the alternate hypothesis
    # ---------------------------

    # Run minimisation and Hesse algorithm
    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    h1_valid = mi.valid
    if not(h1_valid):
        print('Warning: valid minimum NOT FOUND for H1')
        print(mi)
        return 'invalid minimum', 0, 0 # don't continue with test

    # Parameter values for total model fit
    h1_params = list(mi.values)

    # Negative log likelihood for the dataset given the alternate hypothesis 
    h1_nll = mi.fval

    # ---------------------------
    # Run the fit for the null hypothesis
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
    h0_valid = mi.valid
    if not(h0_valid):
        print('Warning: valid minimum NOT FOUND for H0')
        print(mi)
        return 'invalid minimum', 0, 0 # don't continue with test

    # Parameter values for background-only fit
    h0_params = list(mi.values)

    # negative log likelihood for the dataset given the null hypothesis 
    h0_nll = mi.fval

    # ---------------------------
    # Perform Neyman-Pearson Test
    # ---------------------------

    print(f'h0_nll: {h0_nll}, h1_nll: {h1_nll}')
    Z, p_value = neyman_pearson_test(h0_nll, h1_nll)

    # If we get a significance greater than 5, we have 'discovered' the signal
    if Z >= 5:
        discovery = True
    else:
        discovery = False

    if plot:
        fig = plotting(dataset, pdf, h0_params, h1_params)
        return fig, discovery, Z, p_value

    return discovery, Z, p_value


def two_signal_test(dataset, pdf, cdf, starting_params, binned=False, plot=False):
    """
    Perform Neyman-Pearson Hypothesis test for the existance of two distinct signals

    Null hypothesis is just 1 signal. Alternate is 2 signals
    """

    # --------------------
    # Minimisation object
    # --------------------
    if binned:
        
        # Bin the dataset
        bins = int(np.sqrt(len(dataset))) # square root rule for number of bins
        bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)

        # Cost function is binned negative log likelihood
        nll = BinnedNLL(bin_density, bin_edges, cdf)

        # Minimisation object
        mi = Minuit(nll, **starting_params)

    else:
        # Cost function as unbinned negative log likelihood
        nll = UnbinnedNLL(dataset, pdf)
        mi = Minuit(nll, **starting_params)

    # Setting constraints
    mi.limits['f1'] = (0, 0.5)
    mi.limits['f2'] = (0, 0.5)
    mi.limits['mu1'] = (5, 5.6) # the signal should not peak outside of [alpha, beta]
    mi.limits['mu2'] = (5, 5.6)
    mi.limits['lam'] = (0, None) # lambda cannot be negative (otherwise there is no 'decay')
    mi.limits['sigma'] = (0, 0.3) # sigma should not be too wide, and cannot be negative

    # ---------------------------
    # Run the fit for the alternate hypothesis
    # ---------------------------

    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    h1_valid = mi.valid
    if not(h1_valid):
        print('Warning: valid minimum NOT FOUND for H1')
        print(mi)
        return 'invalid minimum', 0, 0 # don't continue with test

    h1_params = list(mi.values)

    # Negative log likelihood for the dataset given the alternate hypothesis 
    h1_nll = mi.fval

    # ---------------------------
    # Run the fit for the null hypothesis
    # ---------------------------
    mi.values['f2'] = 0
    mi.fixed['f2'] = True
    mi.fixed['mu2'] = True

    mi.migrad()
    mi.hesse()

    # If valid minimum is not found, print message
    h0_valid = mi.valid
    if not(h0_valid):
        print('Warning: valid minimum NOT FOUND for H0')
        print(mi)
        return 'invalid minimum', 0, 0 # don't continue with test

    # Parameter values for total model fit
    h0_params = list(mi.values)

    # Negative log likelihood for the dataset given the alternate hypothesis 
    h0_nll = mi.fval

    # ---------------------------
    # Perform Neyman-Pearson Test
    # ---------------------------

    Z, p_value = neyman_pearson_test(h0_nll, h1_nll)

    # If we get a significance greater than 5, we have 'discovered' the signal
    if Z >= 5:
        discovery = True
    else:
        discovery = False

    if plot:
        fig = plotting(dataset, pdf, h0_params, h1_params)
        return fig, discovery, Z, p_value

    return discovery, Z, p_value
