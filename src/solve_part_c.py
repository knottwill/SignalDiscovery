"""
In this script we solve part c by checking that the total PDF
integrates to unity over [5, 5.6] for 1000 random combinations
of parameters. The outcome of the integral is checked to be
approximately equal to 1 (with a relative tolerance of 1e-6).
This is necessary since the integral outcome is a float, which
often results in bad behaviour where the float is not perfectly 
equal to the integer 1 (but is extremely close)
"""

import numpy as np
from scipy.integrate import quad
from pytest import approx

# Import the total PDF
from distributions import total_pdf

# Define upper and lower bounds
alpha = 5
beta = 5.6

# Integrate the PDF for 1000 random combinations of parameters
integrates_to_unity = True
N_combos = 1000
np.random.seed(42)
print(f'Integrating PDF over [{alpha}, {beta}] for {N_combos} random combinations of parameters...\n')
for _ in range(N_combos):

    # Generate parameter values from uniform distribution over an appropriate range
    f = np.random.uniform(0,1)
    lam = np.random.uniform(0,2)
    mu = np.random.uniform(alpha, beta)
    sigma = np.random.uniform(0.1,0.7)

    # Performs integration of total PDF over range [alpha, beta]
    total_probability, error = quad(total_pdf, alpha, beta, args=(f, lam, mu, sigma))

    # Check that integral came out to 1 (with a relative tolerance of 1e-6) 
    relative_tolerance = 1e-6
    if total_probability != approx(1, rel=relative_tolerance):
        integrates_to_unity=False # failed

# Print statements if test was succeeded vs failed
if integrates_to_unity:
    print(f'As expected, the total PDF integrated to unity (with a relative ')
    print(f'tolerance of {relative_tolerance}) on all {N_combos} trials\n')
else:
    print('Failed.')