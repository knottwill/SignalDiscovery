import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

# True parameter values
true_params = {'f': 0.1, 'lam': 0.5, 'mu': 5.28, 'sigma': 0.018}

# Upper & Lower bound of M
alpha = 5
beta = 5.6

# ---------------
# Generating data
# ---------------
from generation import generate_from_total_pdf

N_events= 100000
dataset = generate_from_total_pdf(N_events, **true_params)

# ---------------
# Starting parameters for minimisation
# Our starting parameters are the true parameters + some approprate random shift
# The shift is limited such that we don't start with 'unphysical' parameters
# or with parameters too far away from the true parameters
# ---------------

random_shifts = {
    'f': np.random.uniform(-0.09, 0.5), 
    'lam': np.random.uniform(-0.3, 1), 
    'mu': np.random.uniform(-0.25, 0.3), 
    'sigma': np.random.uniform(-0.01, 0.03)
}

# Creating starting parameters
starting_params = {}
for param in true_params:

    # true parameter + random shift
    starting_params[param] = true_params[param] + random_shifts[param]

    # round to 4 significant figures
    starting_params[param] = float(f'{starting_params[param]:.4}')

# ---------------
# Minimisation Object
# ---------------

# import our total PDF
from distributions import total_pdf

# Cost function is negative log likelihood (with factor 2 so it is chi2 distributed)
nll = UnbinnedNLL(dataset, total_pdf)

# Minimisation object
mi = Minuit(nll, **starting_params)

# ---------------
# Setting constraints to parameters 
# some are physical constaints (eg. sigma > 0) 
# some are just sensible constraints
# ---------------
mi.limits['f'] = (0, 1) # fraction of signal is between 0 and 1
mi.limits['lam'] = (0, None) # lambda cannot be negative (otherwise there is no 'decay')
mi.limits['sigma'] = (0, (beta-alpha)/2) # sigma should not be too wide, and cannot be negative
mi.limits['mu'] = (alpha, beta) # the signal should not peak outside of [alpha, beta]

# ---------------
# Running Minimisation and the error finding algorithms
# ---------------

mi.migrad() # minimisation
mi.hesse() # finds symmetric uncertainty
mi.minos() # finds non-symmetric confidence interval

print(mi)

# ---------------
# Binning and Plotting
# ---------------

# Bin the events
bins = 120
bin_density, bin_edges = np.histogram(dataset, bins=bins, density=True)
counts, _ = np.histogram(dataset, bins=bins, density=False)

# Calculating uncertainty on bin_density
count_uncertainties = np.sqrt(counts) 
bin_widths = np.diff(bin_edges)
bin_err = count_uncertainties / (len(dataset) * bin_widths)

# Calculate bin midpoints
midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fig, ax = plt.subplots()

# Plotting true model, fitting model and the binned data
ax.errorbar(midpoints, bin_density, yerr=bin_err, fmt='.', capsize=1, label='Binned Data')
ax.plot(midpoints, total_pdf(midpoints, **true_params), label='True model', color='red')
ax.plot(midpoints, total_pdf(midpoints, *mi.values), label='Fitted model', color='orange')

ax.set_xlabel('M')
ax.set_ylabel('Density')
ax.set_title(f'{N_events} total events')

ax.legend()

fig.savefig('plots/part_e.png')