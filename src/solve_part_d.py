import matplotlib.pyplot as plt
import numpy as np

# import pdfs
from pdfs import signal_pdf, background_pdf, total_pdf

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

# Evaluate pdfs at 
pdf_signal = signal_pdf(M_values, mu, sigma, alpha, beta)
pdf_background = background_pdf(M_values, lam, alpha, beta)
pdf_total = total_pdf(M_values, f, lam, mu, sigma, alpha, beta)

# Add weightings to signal and background
signal_only = f*pdf_signal
background_only = (1-f)*pdf_background

# Ensure the total pdf is equal to signal + background (with weightings)
assert( np.all(pdf_total == signal_only + background_only) )


####################
## Plotting the PDFs 
#####################
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Creates a figure with two subplots

# First Plot - signal and background have weights applied
axs[0].plot(M_values, pdf_total, label='Total Probability Distribution')
axs[0].plot(M_values, signal_only, label=f'Signal Only (with {f} weighting)', linestyle=':')
axs[0].plot(M_values, background_only, label=f'Background Only (with {1-f} weighting)', linestyle='--')

axs[0].set_xlabel('M')
axs[0].set_ylabel('Probability Density')
axs[0].set_title('Comparison of Probability Distributions (Weighted)')
axs[0].legend()
axs[0].grid(True)

# Second Plot - signal and background are normalised PDFs
axs[1].plot(M_values, pdf_total, label='Total Probability Distribution')
axs[1].plot(M_values, pdf_signal, label='Signal PDF (Normalized)', linestyle=':')
axs[1].plot(M_values, pdf_background, label='Background PDF (Normalized)', linestyle='--')

axs[1].set_xlabel('M')
axs[1].set_ylabel('Probability Density')
axs[1].set_title('Comparison of Probability Distributions (Normalized)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()  # Adjusts the plots to fit into the figure neatly

fig.savefig(f'plots/part_d.png')
