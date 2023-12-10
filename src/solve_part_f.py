"""
In this script we solve part f. This is done in three stages:

*** Finding probability of discovery **
----------------------------------------
For 50 dataset sizes in the range [400, 900] we estimate the 
probability of discovering the signal using the custom function 
'probability_of_discovery' in the discovery.py module. We pass arguments
to specify that 500 example datasets should be generated using the generation
function that generates events from the total PDF, then each dataset should
be subjected to the hypothesis test 'signal_background_test' (from the module
hypothesis_test.py). This 'signal_background_test' function fits a background-only model and a
'signal + background' model to the dataset to serve as the null and alternate
hypotheses, then calculates the p value using the Neyman-Pearson test statistic,
and returns whether the p value is small enough to constitute a 'discovery'.
The 'probability_of_discovery' then estimates the probability of discovery and
uncertainty, using the outcome of all the hypothesis tests, and returns these estimates.

*** Fitting predictive model ***
----------------------------------
A third order polynomial is then fitted to the 'discovery probability (P) vs dataset size (N)' 
data using least squares estimation, and we calculate some goodness-of-fit metrics using the 
custom function 'goodness_of_fit' (metrics: coverage, chi-squared per degree of freedom,
and chi-squared test p value). Note: the dataset sizes N are normalised before the model
is fitted, since helps iminuit estimate the parameters. 

*** Finding critical size, N90 ***
-----------------------------------
We then find the size of the dataset, N90, that corresponds to a 90% probability of
discovering the signal by solving the equation: model(N90) = 0.9 
(where 'model' is the fitted third order polynomial). The uncertainty is estimated
using a Monte Carlo simulation where the parameters of the model are sampled from a
normal distribution (according to the uncertainties found from the least squares fit),
then the equation is solved. The standard deviation of the set of solutions becomes the
uncertainty for N90. This is all done with the custom function 'find_solution' in the 
NP_analysis.py module.
"""

from time import time
import numpy as np
import pickle 
import os
from iminuit import Minuit
from iminuit.cost import LeastSquares

# imports from custom modules
from generation import generate_from_total_pdf
from hypothesis_test import signal_background_test
from discovery import probability_of_discovery
from NP_analysis import goodness_of_fit, find_solution, plot_NP

start = time()
np.random.seed(42)

true_params = {'f': 0.1, 'lam': 0.5, 'mu': 5.28, 'sigma': 0.018}

# Make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# --------------------
# Calculating Discovery Probabilities 
# --------------------

# array storing the data sizes at which we will
# find the probability of discovering the signal
N = np.linspace(400, 900, 50, dtype=int)

P = [] # probability of discovery
P_err = [] # error in probability of discovery
for i, N_events in enumerate(N):

    p, p_err = probability_of_discovery(
        N_events=N_events, # size of dataset
        n_trials=500, # number of datasets to generate and test
        true_params=true_params, # true parameters for total PDF
        generation_func=generate_from_total_pdf, # function to generate datasets
        hypothesis_test=signal_background_test, # hypothesis test to conduct on each dataset
    )

    P.append(p)
    P_err.append(p_err)
    # printing loading messages
    if 2*(i+1)%10 == 0:
        print(f"{2*(i+1)}% complete")
P, P_err = np.array(P), np.array(P_err)

# Save data (just in case)
data_to_save = {'N': N, 'P': P, 'P_err': P_err}
with open('part_f_data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

# ---------------------
# Fit 3rd degree polynomial to data
# ----------------------

# Replace 0 uncertainties with the minimum non-zero uncertainty
# so as to not produce errors in the least squares estimation
P_err[P_err == 0.0] = min(P_err[P_err != 0])

# We normalise N since it helps the fitting
N_norm = (N-np.mean(N))/np.std(N)

# Third degree polynomial model for fitting
def third_degree(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d 

# Least squares fit
cost = LeastSquares(N_norm, P, P_err, third_degree)
starting_params = {'a': 0.01, 'b': -0.05, 'c': 0.1, 'd': 0.9}
mi = Minuit(cost, **starting_params)
mi.migrad()

# Calculating predicted values and pulls
P_pred = third_degree(N_norm, *mi.values)
pull = (P - P_pred)/P_err

# goodness-of-fit evaluation
coverage, p_value, chisq_per_dof = goodness_of_fit(pull)
print(f"coverage of fitted model: {round(100*coverage)}%")
print(f"chi2 test p value: {p_value:.5}")
print(f"chi2 per degree of freedom: {chisq_per_dof:.5}\n")

# ---------------------
# Calculate size of dataset for 90% discovery rate
# ----------------------

initial_guess = (600 - np.mean(N))/np.std(N)
N90_norm, N90_norm_err = find_solution(
    target=0.9, # target probability (0.9 means 90%)
    initial_guess=initial_guess, # initial guess for N90
    model = third_degree, # predictive model
    params=mi.values, # parameter estimates of predictive model
    errors=mi.errors # uncertainties of parameter estimates
)

# converting back to non-normalised sizes
N90 = N90_norm*np.std(N) + np.mean(N)
N90_err = np.std(N)*N90_norm_err
print(f"Size of dataset for 90% discovery rate: {round(N90)} +- {round(N90_err)}")

# Plot results
plot_NP(N, P, P_err, P_pred, N90, filepath="plots/part_f.png")

print(f'\nFinished in {time() - start:.4}s')