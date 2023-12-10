"""
Module to assist with the analysis of the 'probability of discovery', P, vs
'size of dataset', N, data in part f and g
"""
import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from pytest import approx
from scipy.optimize import fsolve

def goodness_of_fit(pull):
    """
    Calculate various goodness-of-fit metrics using the Pulls

    This function calculates the coverage of a model on the data
    and performs a chi2 goodness-of-fit test. It returns the 
    coverage (as a decimal) and the p-value and chi2 per degree of 
    freedom as calculated by the chi2 test.

    Parameters
    ------------
    pull: np.ndarray
        An array containing the 'pulls' of the fitting model on the data.

    Returns
    ----------
    tuple containing goodness-of-fit metrics. The elements are:

        - coverage: Fraction of the model's predictions which lie
        within the uncertainty of the measurement. (Should be close to 68.3%)

        - p_value: p value of the chi2 test. This is the probability of observing
        a chi2 as extreme or more extreme than the value measured, assuming the model
        is correct.
        
        - chisq_per_dof: the chi2 per degree of freedom 
        (should be close to 1)
    """

    # coverage
    n_covers = np.count_nonzero(np.abs(pull) < 1)
    coverage = n_covers/len(pull)

    # chi squared test
    chisq = np.sum( pull**2 )
    ndof = len(pull) - 4 # number of degrees of freedom
    chisq_per_dof = chisq/ndof
    p_value = 1 - chi2.cdf(chisq, ndof)

    return coverage, p_value, chisq_per_dof


def find_solution(target, initial_guess, model, params, errors):
    """
    Find solution to equation model(x) = target

    This function finds the value of x for which model(x) = target, 
    using scipy.optimize.fsolve. It finds an error for this value via a 
    Monte Carlo simulation, by randomly sampling the model parameters 
    according to their uncertainties and solving the equation for each
    set of parameters. This set of solutions is then used to estimate 
    the uncertainty. 

    Parameters
    ------------
        target: float
            The target value that the model should evaluate to at the solution
        initial_guess: float
            An initial guess for the solution
        model: function
            The model which maps the solution to the target
        params: list
            The parameters of the model
        errors: list
            The errors in the parameters of the model

    Returns
    -----------
    tuple
        Tuple containing the solution and it's uncertainty
    """

    # define equation to be solved
    def equation(x, *params):
        return model(x, *params) - target
    
    solution = fsolve(equation, initial_guess, args=tuple(params))[0]

    assert model(solution, *params) == approx(target), "Solution not found"
    
    # randomly sample combinations of parameters 
    # according to their uncertainties
    param_samples = []
    num_simulations = 1000
    np.random.seed(42)
    for i, param in enumerate(params):
        samples = np.random.normal(param, errors[i], num_simulations)
        param_samples.append(samples)

    param_samples = zip(*param_samples)

    # solving equation for each set of parameters
    solutions = []
    for param_sample in param_samples:
        solutions.append(fsolve(equation, initial_guess, args=tuple(param_sample))[0])
    solutions = np.array(solutions)

    # calculate uncertainty as the standard deviation of the solution measurements
    err = np.std(solutions)

    return solution, err


def plot_NP(N, P, P_err, P_pred, N90, filepath=False):
    """
    Plot the probability of discovery vs dataset size data

    This function makes a plot of the probability of discovery vs the dataset
    size data, and overlays the fitted model. It shows the solution for the
    critical dataset size (for a 90% discovery rate) via axis lines on the plot.
    It also plots the pulls for a visual illustration of goodness-of-fit. 
    """

    pull = (P - P_pred)/P_err

    fig, ax = plt.subplots(2, 2, figsize=(6.4,6.4), sharex='col', sharey='row',
        gridspec_kw=dict(hspace=0, wspace=0, height_ratios=(3,1), width_ratios=(7,1)))

    # probability vs dataset size plot
    ax[0,0].errorbar( N, P, yerr=P_err, capsize=2, fmt='.', c='black')
    ax[0,0].plot(N, P_pred, label='Least Squares Fit')
    ax[0,0].set_ylabel('Probability of discovery')
    ax[0,0].axhline(y=0.9, color='green', linestyle='--')
    ax[0,0].axvline(x=N90, color='green', linestyle='--')

    # pull vs dataset size plot
    ax[1,0].errorbar( N, pull, yerr=np.ones_like(N), capsize=2, fmt='.', c='black')
    ax[1,0].plot(N, np.zeros_like(N), color='r', linestyle='--')
    ax[1,0].set_xlabel('Size of Dataset, $N$')
    ax[1,0].set_ylabel('Pull')
    ax[1,0].set_yticks([-2,0,2])

    # histogram plot on the right hand side of the pull plot
    ax[0,1].set_visible(False)
    ax[1,1].hist(pull, bins=10, range=(-3,3), density=True, alpha=0.7, orientation='horizontal')
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].spines[['top','bottom','right']].set_visible(False)
    ax[1,1].tick_params( which='both', direction='in', axis='y', right=False, labelcolor='none')
    # standard normal distribution overlaid on top of histogram
    xp = np.linspace(-3,3,100)
    ax[1,1].plot( norm.pdf(xp), xp, 'r-', alpha=0.5 )

    # tight x-axis
    ax[0,0].autoscale(enable=True, tight=True, axis='x')

    ax[0,0].legend()

    fig.align_ylabels()

    if filepath:
        fig.savefig(filepath)