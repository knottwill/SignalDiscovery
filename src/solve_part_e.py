"""
In this script we solve part e by generating a sample of 100,000
events, then fit a the total PDF to the sample using maximum likelihood
estimation with iminuit. As starting parameters, we add a random shift 
to the true parameter values of up to 30% of their absolute value. 
The uncertainties are calculated by iminuit as the minimum variance bound. 
We then bin and plot the data, where the uncertainties are given by the 
square root of the bin counts, then scaled to be the uncertainty of the 
bin density rather than bin count. The signal, background and total
PDFs are overlaid over the data using the estimated parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.style as mplstyle
import os

# imports from custom modules (generation function and PDFs)
from generation import generate_from_total_pdf
from distributions import total_pdf, signal_pdf, background_pdf

mplstyle.use('src/mphil.mplstyle')

# True parameter values
true_params = {'f': 0.1, 'lam': 0.5, 'mu': 5.28, 'sigma': 0.018}

# Upper & Lower bound of M
alpha = 5
beta = 5.6


# ---------------
# Generating 100K events
# ---------------
from generation import generate_from_total_pdf

np.random.seed(42)
N_events= 100000
dataset = generate_from_total_pdf(N_events, **true_params)


# ------------------------------
# Starting parameters for minimisation
# We generate starting parameters by adding random shifts 
# to the true parameters of up to 30% of their absolute value
# (the parameters are rounded to 4 significant figures)
# ------------------------------
starting_params = {}
for param in true_params:
    # generating random shift
    max_shift = 0.3*np.abs(true_params[param])
    random_shift = np.random.uniform(-max_shift, max_shift)

    # adding random shift
    starting_params[param] = true_params[param] + random_shift

    # rounding to 4 sig. fig.
    starting_params[param] = float(f'{starting_params[param]:.4}')


# ---------------
# Fitting PDF using maximum likelihood estimation
# (Minimisation of negative log likelihood)
# ---------------

# Cost function is negative log likelihood
nll = UnbinnedNLL(dataset, total_pdf)

# Minimisation object
mi = Minuit(nll, **starting_params)

# Setting constraints
mi.limits['f'] = (0, 1) # fraction of signal is between 0 and 1
mi.limits['lam'] = (0, None) # lambda cannot be negative (otherwise there is no 'decay')
mi.limits['sigma'] = (0, (beta-alpha)/2) # sigma should not be too wide, and cannot be negative
mi.limits['mu'] = (alpha, beta) # the signal should not peak outside of [alpha, beta]

# Running Minimisation and the error finding algorithms
mi.migrad() # minimisation
mi.hesse() # finds symmetric uncertainty (min variance bound)
mi.minos() # finds non-symmetric confidence interval

assert mi.valid
print(mi)
print('')
print('Parameter estimates:')
print(f"f: {mi.values['f']} +- {mi.errors['f']}")
print(f"lam: {mi.values['lam']} +- {mi.errors['lam']}")
print(f"mu: {mi.values['mu']} +- {mi.errors['mu']}")
print(f"sigma: {mi.values['sigma']} +- {mi.errors['sigma']}")

# ---------------
# Binning and Plotting Results
# ---------------

# Bin the events
bins = 120
bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)
bin_counts, _ = np.histogram(dataset, bins=bins, density=False)
midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # bin midpoints

# Calculating uncertainty on bin_density
count_uncertainties = np.sqrt(bin_counts) 
bin_widths = np.diff(bin_edges)
density_uncertainties = count_uncertainties / (len(dataset) * bin_widths)

# Evaluate the PDFs
pdf_total = total_pdf(midpoints, *mi.values)
weighted_signal = mi.values['f'] * signal_pdf(midpoints, mi.values['mu'], mi.values['sigma'])
weighted_background = (1-mi.values['f']) * background_pdf(midpoints, mi.values['lam'])

fig, ax = plt.subplots()

# Plotting fitted model, signal and background with binned data
ax.errorbar(midpoints, bin_density, yerr=density_uncertainties, fmt='.', color='black', capsize=1, label='Binned Data')
ax.plot(midpoints, pdf_total, label='Fitted PDF', color='#1f77b4')
ax.plot(midpoints, weighted_signal, label='Signal', color='#ff7f0e', linestyle=':')
ax.plot(midpoints, weighted_background, label='Background', color='#2ca02c', linestyle='--')
ax.grid(True)
ax.set_xlabel('M')
ax.set_ylabel('Density')
ax.legend()

# Make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

fig.savefig('plots/part_e.png')
print('')
print('Plot saved in plots/part_e.png')