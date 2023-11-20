import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
import seaborn as sns
from time import time

# Upper & lower bound
alpha = 5
beta = 5.6

# True parameter values
f = 0.1
lam = 0.5 
mu = 5.28  
sigma = 0.018 

###############################
# Using the inverse CDF method, I will generate
# 90K events from background and 10K events from singal
# which is the same as generating 100K events from the total PDF
###############################

# number of (signal and background) events
N_events = 100000
N_signal_events = int( N_events*f )
N_background_events = int( N_events*(1-f) )

########################
# Lower and upper bounds for the probability to feed
# into the inverse CDF function can be found by
# evaulating the CDFs at alpha and beta
########################
lower_p_signal = norm.cdf(alpha, loc=mu, scale=sigma)
upper_p_signal = norm.cdf(beta, loc=mu, scale=sigma)
lower_p_background = expon.cdf(alpha, scale=1/lam)
upper_p_background = expon.cdf(beta, scale=1/lam)

# Measure time to generate data
t0 = time()

# Generating events using inverse CDF ('ppf')
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

# Print time
t1 = time()
print(f'Generated {N_events} events in {t1-t0:.4}s')

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


