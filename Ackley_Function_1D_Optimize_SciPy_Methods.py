# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:41:12 2024

@author: Punit Bandi

Classic Optimization Benchmark Problem "Ackley Function" with 1 Dimension
"""

import math
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy import optimize

def ackley_function(x):
    """
    Compute the Ackley function for a given input x (1-dimensional).
    
    Parameters:
    x (float): Input value.
    
    Returns:
    float: Value of the Ackley function at x.
    """
    a = 20
    b = 0.2
    c = 2 * math.pi
    
    x = np.array(x, ndmin=1)  # Ensure x is treated as a numpy array
    term1 = -a * math.exp(-b * math.sqrt(x**2))
    term2 = -math.exp(math.cos(c * x))
    
    return term1 + term2 + a + math.exp(1)

# Define input range
inputs = np.arange(-100, 101, 1)  # Range from -100 to 100 with interval 1

# Compute Ackley function for each input
outputs = np.array([ackley_function(x) for x in inputs])

# Print inputs and outputs
#print("Inputs:", inputs)
#print("Outputs:", outputs)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(inputs, outputs, label='Ackley Function for 1 Dimension', color='blue', linewidth=2)

# Adding labels and title
plt.title("Plot of y vs x", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)

# Adding grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Display the plot
plt.show()

# Optimization
x0 = [30] # Initial guess
b = [(-100,100)] # Bounds as a list of tuples

# Univariate Function Minimizer Method 'brent'
sol_brent = minimize_scalar(ackley_function,method='brent')
print("Optimized Objective Value using Brent Method:", sol_brent.fun)
print("Input Value for achieving minimum objective using Brent Method:", sol_brent.x)
print("Total function calls needed to find optima", sol_brent.nfev)

# Global optimization using simplicial homology global optimisation (shgo) algorithm
sol_shgo = optimize.shgo(ackley_function,b)
print("Optimized Objective Value using SHGO:", sol_shgo.fun)
print("Input Value for achieving minimum objective using SHGO:", sol_shgo.x)
print("Total function calls needed to find optima", sol_shgo.nfev)

# Global optimization using Dual Annealing Optimization algorithm
sol_DA = optimize.dual_annealing(ackley_function,b)
print("Optimized Objective Value using Dual Annealing:", sol_DA.fun)
print("Input Value for achieving minimum objective using Dual Annealing:", sol_DA.x)
print("Total function calls needed to find optima", sol_DA.nfev)