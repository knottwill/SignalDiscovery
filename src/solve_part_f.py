from find_NP import generate_NP_data
from find_N90 import find_N90
from time import time
import pickle 

start = time()

N, P, P_err = generate_NP_data(n_datapoints=50, n_attempts=500, N_range=(400,900))

# Save data (just in case)
data_to_save = {'N': N, 'P': P, 'P_err': P_err}
with open('PN_data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)


N90, N90_err, coverage, p_value, Z = find_N90(N, P, P_err, plot_filepath='plots/part_f.png')

print(f"Result = {N90} +- {N90_err}\n")
print(f"Finished in {time() - start: .4}s")
print(f"Line fitted via least squares had a coverage of {coverage}\n")
print(f"Goodness of fit test results: p value = {p_value}, Z = {Z}")