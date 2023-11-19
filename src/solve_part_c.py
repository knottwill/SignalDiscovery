import numpy as np
from scipy.integrate import quad
from pytest import approx

# Import the pdfs
from pdfs import signal_pdf, background_pdf, total_pdf

# Define upper and lower bounds
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