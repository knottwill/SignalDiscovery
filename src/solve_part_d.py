import matplotlib.pyplot as plt
import numpy as np
from pdfs import signal_pdf, background_pdf, total_pdf

alpha = 5
beta = 5.6

f = 0.1
lam = 0.5 
mu = 5.28  
sigma = 0.018 

M_values = np.linspace(alpha, beta, 1000)

signal_only = f*signal_pdf(M_values, mu, sigma, alpha, beta)
background_only = (1-f)*background_pdf(M_values, lam, alpha, beta)
pdf_total = total_pdf(M_values, f, lam, mu, sigma, alpha, beta)

assert( np.all(pdf_total == signal_only + background_only) )

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(M_values, pdf_total, label='Total Probability Distribution')
ax.plot(M_values, signal_only, label=f'Signal Only (with {f} weighting)', linestyle=':')
ax.plot(M_values, background_only, label=f'Background Only (with {1-f} weighting)', linestyle='--')

ax.set_xlabel('M')
ax.set_ylabel('Probability Density')
ax.set_title('Comparison of Probability Distributions')

ax.legend()
ax.grid(True)

fig.savefig(f'plots/part_d.png')