from time import time
import numpy as np
import pickle 
import os
from iminuit import Minuit
from iminuit.cost import LeastSquares

from generation import generate_from_two_signal_pdf
from hypothesis_test import two_signal_test
from discovery import probability_of_discovery
from NP_analysis import goodness_of_fit, find_solution, plot_NP

start = time()
np.random.seed(42)

true_params = {"f1": 0.1, "f2": 0.05, "lam": 0.5, "mu1": 5.28, "mu2": 5.35, "sigma": 0.018}

# Make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# --------------------
# Calculating Discovery Probabilities 
# --------------------

# array storing the data sizes at which we will
# find the probability of discovering the signal
N = np.linspace(1500, 3000, 50, dtype=int)

P = [] # probability of discovery
P_err = [] # error in probability of discovery
for i, N_events in enumerate(N):

    p, p_err = probability_of_discovery(
        N_events=N_events, 
        n_trials=300,
        true_params=true_params,
        generation_func=generate_from_two_signal_pdf,
        hypothesis_test=two_signal_test,
    )

    P.append(p)
    P_err.append(p_err)
    # printing loading messages
    if 2*(i+1)%10 == 0:
        print(f"{2*(i+1)}% complete")
P, P_err = np.array(P), np.array(P_err)

# Save data (just in case)
data_to_save = {'N': N, 'P': P, 'P_err': P_err}
with open('part_g_data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

# import pandas as pd
# filepath = 'part_g_data.pkl'
# data = pd.read_pickle(filepath)

# N = data['N']
# P = data['P']
# P_err = data['P_err']

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
starting_params = {'a': 0.004, 'b': -0.04, 'c': 0.2, 'd': 0.8}
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

initial_guess = (2600 - np.mean(N))/np.std(N)
N90_norm, N90_norm_err = find_solution(
    target=0.9,
    initial_guess=initial_guess,
    model = third_degree, 
    params=mi.values,
    errors=mi.errors
)

# converting back to non-normalised sizes
N90 = N90_norm*np.std(N) + np.mean(N)
N90_err = np.std(N)*N90_norm_err
print(f"Size of dataset for 90% discovery rate: {round(N90)} +- {round(N90_err)}")

# Plot results
plot_NP(N, P, P_err, P_pred, N90, filepath="plots/part_g.png")

print(f'\nFinished in {time() - start:.4}s')
