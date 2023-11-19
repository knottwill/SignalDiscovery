import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, expon
import seaborn as sns

from time import time

# Import pdfs
from pdfs import signal_pdf, background_pdf, total_pdf

# Upper & lower bound
alpha = 5
beta = 5.6

# True parameter values
f = 0.1
lam = 0.5 
mu = 5.28  
sigma = 0.018 

# Im pretty sure that if I generate 90K events from background
# and 10K events from signal
# then overall this will be the same as generating 100K events from total

# number of events
N_events = 10000000
N_signal_events = int(N_events*f)
N_background_events = int(N_events*(1-f))

# Im pretty sure this works to make it right
# since all we care about is 'relative' probability density, 
# which is valid even if we don't normalise
lower_p_signal = norm.cdf(alpha, loc=mu, scale=sigma)
upper_p_signal = norm.cdf(beta, loc=mu, scale=sigma)
lower_p_background = expon.cdf(alpha, scale=1/lam)
upper_p_background = expon.cdf(beta, scale=1/lam)

t0 = time()

# Generating events using inverse cdf (ppf)
signal_events = norm.ppf(
    q=np.random.uniform(lower_p_signal, upper_p_signal, N_signal_events), 
    loc=mu, 
    scale=sigma
    )

background_events = expon.ppf(
    q=np.random.uniform(lower_p_background, upper_p_background, N_background_events), 
    scale=1/lam
    )

total_events = np.concatenate((signal_events, background_events))

t1 = time()

print(f'Time to generate {N_events} events: {t1-t0:.4}s')

# Bin the events
bins = 50
hist_total, bin_edges_total = np.histogram(total_events, bins=bins)

# Calculate bin midpoints
midpoints_total = 0.5 * (bin_edges_total[1:] + bin_edges_total[:-1])

fig, ax = plt.subplots()

# Plotting bin count vs midpoints
ax.plot(midpoints_total, hist_total, label='Events generated from Total PDF', marker='o')

ax.set_xlabel('M')
ax.set_ylabel('Bin Count')
ax.set_title(f'{N_events} total events')

ax.legend()


fig.savefig('plots/part_e.png')


