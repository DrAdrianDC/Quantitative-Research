# Bayesian optimization


import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

# Define the objective function to minimize
def objective_function(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

# Define the search space
space = [Real(0.0, 5.0, name='x1'),  # Continuous space for x1
         Real(0.0, 5.0, name='x2')]  # Continuous space for x2

# Perform Bayesian Optimization
result = gp_minimize(objective_function,      # The function to minimize
                     space,                   # The search space
                     n_calls=20,              # The number of evaluations
                     random_state=42)         # Random state for reproducibility

# Print the best parameters and the corresponding minimum value
print("Best parameters: x1 = {:.4f}, x2 = {:.4f}".format(result.x[0], result.x[1]))
print("Minimum value: {:.4f}".format(result.fun))

# Plot convergence
plot_convergence(result)

plt.show()



