from generation import generate_from_two_signal_pdf
from hypothesis_test import two_signal_test
from discovery import probability_of_discovery
from distributions import two_signal_cdf, two_signal_pdf
from critical_size import find_N90
from time import time
import numpy as np
import pickle 

start = time()
np.random.seed(42)

true_params = {"f1": 0.1, "f2": 0.05, "lam": 0.5, "mu1": 5.28, "mu2": 5.35, "sigma": 0.018}

# --------------------
# Getting NP data
# --------------------

# array storing the data sizes at which we will
# find the probability of discovering the signal
N = np.linspace(1500, 3000, 50, dtype=int)

P = [] # probability of discovery
P_err = [] # error in probability of discovery
for N_events in N:

    p, p_err = probability_of_discovery(
        N_events=N_events, 
        n_attempts=100,
        true_params=true_params,
        generation_func=generate_from_two_signal_pdf,
        hypothesis_test=two_signal_test,
        pdf = two_signal_pdf,
        cdf = two_signal_cdf
    )

    P.append(p)
    P_err.append(p_err)
    print(f"Sample size (N_events)={N_events}, probability of discovery={P[-1]} +- {P_err[-1]}")
P, P_err = np.array(P), np.array(P_err)

# Save data (just in case)
data_to_save = {'N': N, 'P': P, 'P_err': P_err}
with open('PN_data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

# -----------------------------
# Finding N90
# -----------------------------

N90, N90_err, coverage, p_value, Z = find_N90(N, P, P_err, plot_filepath='plots/part_g.png')

print(f"Result = {N90} +- {N90_err}\n")
print(f"Finished in {time() - start: .4}s")
print(f"Line fitted via least squares had a coverage of {coverage}\n")
print(f"Goodness of fit test results: p value = {p_value}, Z = {Z}")