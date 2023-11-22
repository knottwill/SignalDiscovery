"""
This module contains the code to find N_90 given N, P, P_err
"""
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def plot_NP(N, P, P_err, P_pred, filepath=False):

    pull = (P - P_pred)/P_err

    # plot with the original fit
    fig, ax = plt.subplots(2, 2, figsize=(6.4,6.4), sharex='col', sharey='row',
        gridspec_kw=dict(hspace=0, wspace=0, height_ratios=(3,1), width_ratios=(7,1)))

    # top figure 

    ax[0,0].errorbar( N, P, yerr=P_err, capsize=2, fmt='.', c='black')
    ax[0,0].plot(N, P_pred, label='Fitted line')
    # ax[0,0].errorbar(N, P, yerr=P_err, fmt='o', label='Data')
    ax[0,0].set_ylabel('P')

    # bottom figure residuals
    ax[1,0].errorbar( N, pull, yerr=np.ones_like(N), capsize=2, fmt='.', c='black')
    # ax[1,0].errorbar(N, pull, np.ones_like(N), fmt='o', ecolor='lightgray', alpha=0.5)
    # borrom figure flat line
    ax[1,0].plot(N, np.zeros_like(N), color='r', linestyle='--')
    ax[1,0].set_xlabel('$N$')
    ax[1,0].set_ylabel('Pull')

    # right figure
    ax[0,1].set_visible(False)
    ax[1,1].hist(pull, bins=10, range=(-3,3), density=True, alpha=0.7, orientation='horizontal')
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].spines[['top','bottom','right']].set_visible(False)
    ax[1,1].tick_params( which='both', direction='in', axis='y', right=False, labelcolor='none')
    xp = np.linspace(-3,3,100)
    ax[1,1].plot( norm.pdf(xp), xp, 'r-', alpha=0.5 )

    # tight x-axis
    ax[0,0].autoscale(enable=True, tight=True, axis='x')

    ax[0,0].legend()

    fig.align_ylabels()

    if filepath:
        fig.savefig(filepath)


def find_N90(N, P, P_err, plot_filepath=False):

    def third_degree(x, a, b, c, d):
        return a*(x**3) + b*(x**2) + c*x + d 

    # Replace 0 uncertainties with 0.00001 so as to 
    # not produce errors in the least squares estimation
    P_err[P_err == 0.0] = 0.00001

    # ------------------
    # should maybe use starting values
    # ------------------
    params, cov = curve_fit(third_degree, N, P, sigma=P_err, absolute_sigma=True)

    a, b, c, d = params
    a_err, b_err, c_err, d_err = np.sqrt(np.diag(cov))

    P_pred = third_degree(N, *params)

    plot_NP(N, P, P_err, P_pred, filepath=plot_filepath)

    # ------------------
    # Finding % that fitted line goes through uncertainties
    # ------------------

    pull = (P - P_pred)/P_err

    n_intercepts = np.count_nonzero(np.abs(pull) < 1)
    coverage = n_intercepts/len(pull)
    print(f'Coverage: {coverage}')

    # ------------------
    # Goodness of fit test
    # ------------------

    chisq = np.sum( pull**2 )
    ndof = len(N) - 4 # number of degrees of freedom
    p_value = 1 - chi2.cdf(chisq, ndof)

    # --------------
    # shld this be a one sided or two sided test?
    # --------------
    Z = chi2.ppf(1-p_value,1)**0.5 # significance

    print(f'p value: {p_value}, Z: {Z}')

    # ------------------
    # Find N_90
    # ------------------