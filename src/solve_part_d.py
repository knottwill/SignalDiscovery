"""
In this file we solve part d. We plot the signal, background and total 
distributions with the true parameter values in two ways. In the first 
plot we include the weightings on signal and background such that only 
the total PDF is properly normalised but it is visually clear that the 
combination of the signal-only and background-only models makes up the 
total PDF. In the second plot, we properly normalise all distributions, 
so that we are seeing the 'True PDFs' of the signal-only, background-only 
and total PDFs.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

mplstyle.use('src/mphil.mplstyle')

# import pdfs
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


# ---------------------
# Plotting the weighted PDFs 
# ---------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 

# First Plot - signal and background have weights applied
# such that we can visually see that total = signal + background
ax.plot(M_values, pdf_total, label='Total Probability Distribution')
ax.plot(M_values, weighted_signal, label=f'Signal Only (with {f} weighting)', linestyle=':')
ax.plot(M_values, weighted_background, label=f'Background Only (with {1-f} weighting)', linestyle='--')
ax.set_xlabel('M')
ax.set_ylabel('Probability Density')
ax.set_title('Probability Distributions (Weighted)')
ax.legend()
ax.grid(True)

plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
fig.savefig(f'plots/part_d_weighted.png')

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
ax.set_title('Probability Distributions (Properly Normalized)')
ax.legend()
ax.grid(True)

plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
fig.savefig(f'plots/part_d_normalised.png')
