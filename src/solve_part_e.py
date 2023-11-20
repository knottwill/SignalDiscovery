import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm, expon

# import function to generate data
from generation import generate_from_total_pdf

# generate 100000 events
N_events= 100000
dset = generate_from_total_pdf(N_events, f=0.1, lam=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6)

###############################################
# The models imported from pdfs.py were causing errors
# I redefine the total_pdf as a temporary solution
# GOTTA GO BACK AND CHANGE THIS
###############################################
def model(M, f, lam, mu, sigma):
    total_prob = norm.cdf(x=5.6, loc=mu, scale=sigma) - norm.cdf(x=5, loc=mu, scale=sigma)
    signal = f*norm.pdf(x=M, loc=mu, scale=sigma)/total_prob

    total_prob = expon.cdf(x=5.6, scale=1/lam) - expon.cdf(x=5, scale=1/lam)
    background = (1 - f)*expon.pdf(x=M, scale=1/lam)/total_prob

    return signal + background

# (unbinned) negative log likelihood as the cost function
nll = UnbinnedNLL(dset , model)

# Minimisation object
# Passing random starting values for the parameters
alpha, beta = 5, 5.6
mi = Minuit(
    fcn = nll,
    f = np.random.uniform(0,1),
    lam = np.random.uniform(0,2),
    mu = np.random.uniform(alpha, beta),
    sigma = np.random.uniform(0.1,0.7),
)

# Minimise the cost function
mi.migrad()

# Hesse algorithm
mi.hesse()

# print the fit result
print(mi)


