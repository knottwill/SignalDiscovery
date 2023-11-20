import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
import seaborn as sns
from time import time

# import function to generate data
from generation import generate_from_total_pdf

# generate 100000 events
N_events= 100000
total_events = generate_from_total_pdf(N_events, f=0.1, lam=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6)

# Bin the events
bins = 50
bin_counts, bin_edges = np.histogram(total_events, bins=bins)

# Calculate bin midpoints
midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fig, ax = plt.subplots()

# Plotting bin count vs midpoints
ax.plot(midpoints, bin_counts, label='Events generated from Total PDF', marker='o')

ax.set_xlabel('M')
ax.set_ylabel('Bin Count')
ax.set_title(f'{N_events} total events')

ax.legend()

fig.savefig('plots/part_e.png')


