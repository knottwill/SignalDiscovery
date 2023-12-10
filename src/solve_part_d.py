"""
In this script we solve part d. We plot the signal, background and total 
distributions with the true parameter values in two ways. 
- In the first plot we include the weightings on signal and background such t
hat only the total PDF is properly normalised but it is visually clear that 
the total = signal + background
- In the second plot, we properly normalise all distributions, 
so that we are seeing the 'True PDFs' of the signal-only, background-only 
and total PDFs.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import os

mplstyle.use('src/mphil.mplstyle')

# import PDFs (from custom module)
from distributions import signal_pdf, background_pdf, total_pdf

# Upper & lower bounds
alpha = 5
beta = 5.6

# True values of parameters
f = 0.1
lam = 0.5 
mu = 5.28  
sigma = 0.018 

# Generate values of M in [alpha, beta]
M_values = np.linspace(alpha, beta, 1000)

# Evaluate normalised PDFs at all M_values
pdf_signal = signal_pdf(M_values, mu, sigma)
pdf_background = background_pdf(M_values, lam)
pdf_total = total_pdf(M_values, f, lam, mu, sigma)

# Add weightings to signal and background
weighted_signal = f*pdf_signal
weighted_background = (1-f)*pdf_background

# Ensure the total pdf is equal to signal + background (with weightings)
assert np.all(pdf_total == weighted_signal + weighted_background) 

# Make plots/ directory if it doesn't already exist
if not os.path.exists('plots/'):
    os.makedirs('plots/')

# ---------------------
# Plotting the weighted PDFs 
# ---------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 

# First Plot - signal and background have weights applied
# such that we can visually see that total = signal + background
ax.plot(M_values, pdf_total, label='Total Probability Distribution')
ax.plot(M_values, weighted_signal, label=f'Signal (with {f} weighting)', linestyle=':')
ax.plot(M_values, weighted_background, label=f'Background (with {1-f} weighting)', linestyle='--')
ax.set_xlabel('M')
ax.set_ylabel('Probability Density')
ax.legend()
ax.grid(True)

plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
fig.savefig(f'plots/part_d_weighted.png')
print('First plot saved in plots/part_d_weighted.png')

# ---------------------
# Plotting the normalised PDFs 
# ---------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 

# Second Plot - signal and background are normalised PDFs
# such that all distributions are properly normalised PDFs and integrate to unity
ax.plot(M_values, pdf_total, label='Total Probability Distribution')
ax.plot(M_values, pdf_signal, label='Signal PDF (Normalized)', linestyle=':')
ax.plot(M_values, pdf_background, label='Background PDF (Normalized)', linestyle='--')
ax.set_xlabel('M')
ax.set_ylabel('Probability Density')
ax.legend()
ax.grid(True)

plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
fig.savefig(f'plots/part_d_normalised.png')
print('Second plot saved in plots/part_d_normalised.png')
