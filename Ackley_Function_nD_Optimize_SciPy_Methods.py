# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:41:12 2024

@author: Punit Bandi

Classic Optimization Benchmark Problem "Ackley Function" with n Dimensions
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import optimize


def ackley_function(x):
    """
    Compute the Ackley function for a given input x (n-dimensional).
    
    Parameters:
    x (array-like): Input vector (n-dimensional).
    
    Returns:
    float: Value of the Ackley function at x.
    """
    a = 20
    b = 0.2
    c = 2 * math.pi
    x = np.array(x)  # Ensure x is a numpy array, not a list
    d = len(x)
    
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    
    return term1 + term2 + a + np.exp(1)

# Define input range
x1 = np.arange(-32, 33, 1)  # Range from -32 to 32 with interval 1
x2 = np.arange(-32, 33, 1)  # Range from -32 to 32 with interval 1

# Create meshgrid for the 2D grid of inputs
X1, X2 = np.meshgrid(x1, x2)

# Compute Ackley function for each point in the grid
Z = np.zeros_like(X1)  # Initialize the Z array with the same shape as X1 and X2

# Compute the Ackley function for each (x1, x2) pair
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = ackley_function([X1[i, j], X2[i, j]])

# Plotting the 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, X2, Z, cmap='terrain')

# Add labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title('3D Surface Plot of 2D Ackley Function')

# Show the plot
plt.show()

# Plotting the heatmap
plt.figure(figsize=(10, 7))
plt.imshow(Z, extent=[x1.min(), x1.max(), x2.min(), x2.max()], origin='lower', cmap='plasma', interpolation='bicubic', aspect='auto')
plt.colorbar(label='Ackley Function Value (Z)')
plt.title("Heatmap of 2D Ackley Function")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Optimization
# Define bounds for each dimension
dim = 2
x0 = np.full(dim,10) # 
bounds = [(-32, 32)] * dim  # Typical bounds for Ackley function in optimization problems

# Optimization using Nelder Mead Simplex Algorithm
sol_NM = minimize(ackley_function, x0, method='nelder-mead')
print("Optimized Objective Value for",dim,"dimensional Ackley function using Nelder-Mead:", "{:.2e}".format(sol_NM.fun))
print("Input Value for achieving minimum objective using Nelder-Mead:", [f"{x:.2e}" for x in sol_NM.x])
print("Total function calls needed to find optima", sol_NM.nfev)

# Global optimization using DIRECT algorithm
sol_DIRECT = optimize.direct(ackley_function, bounds)
print("Optimized Objective Value for",dim,"dimensional Ackley function using DIRECT method:", "{:.2e}".format(sol_DIRECT.fun))
print("Input Value for achieving minimum objective using DIRECT:", [f"{x:.2e}" for x in  sol_DIRECT.x])
print("Total function calls needed to find optima", sol_DIRECT.nfev)

# Global optimization using simplicial homology global optimisation (shgo) algorithm
sol_shgo = optimize.shgo(ackley_function,bounds)
print("Optimized Objective Value for",dim,"dimensional Ackley function using SHGO:", "{:.2e}".format(sol_shgo.fun))
print("Input Value for achieving minimum objective using SHGO:", [f"{x:.2e}" for x in sol_shgo.x])
print("Total function calls needed to find optima", sol_shgo.nfev)

# Global optimization using Dual Annealing Optimization algorithm
sol_DA = optimize.dual_annealing(ackley_function,bounds)
print("Optimized Objective Value for",dim,"dimensional Ackley function using Dual Annealing:", "{:.2e}".format(sol_DA.fun))
print("Input Value for achieving minimum objective using Dual Annealing:", [f"{x:.2e}" for x in sol_DA.x])
print("Total function calls needed to find optima", sol_DA.nfev)

# Global optimization using Differential Evolution Optimization algorithm
sol_DE = optimize.differential_evolution(ackley_function,bounds)
print("Optimized Objective Value for",dim,"dimensional Ackley function using Differential Evolution:", "{:.2e}".format(sol_DE.fun))
print("Input Value for achieving minimum objective using Differential Evolution:", [f"{x:.2e}" for x in sol_DE.x])
print("Total function calls needed to find optima", sol_DE.nfev)

# Global optimization using Basin Hopping algorithm
sol_BasinHop = optimize.basinhopping(ackley_function, x0, niter=100, stepsize=0.5)
print("Optimized Objective Value for",dim,"dimensional Ackley function using Basin Hopping:","{:.2e}".format(sol_BasinHop.fun))
print("Input Value for achieving minimum objective using Basin Hopping:", [f"{x:.2e}" for x in sol_BasinHop.x])
print("Total function calls needed to find optima", sol_BasinHop.nfev)

# Solve optimization by increasing dimension size up to MaxDim
MaxDim = 10
NumDim = np.array([])
Objective_DIRECT = np.array([])
FunCall_DIRECT = np.array([])
Objective_shgo = np.array([])
FunCall_shgo = np.array([])
Objective_DA = np.array([])
FunCall_DA = np.array([])
Objective_DE = np.array([])
FunCall_DE = np.array([])
Objective_BasinHop = np.array([])
FunCall_BasinHop = np.array([])

for i in range(MaxDim):
    dim=i+1
    if dim>1:
        x0 = np.full(dim,10) # 
        bounds = [(-32, 32)] * dim  # Typical bounds for Ackley function in optimization problems
        
        NumDim = np.append(NumDim, dim)
        sol_DIRECT = optimize.direct(ackley_function, bounds)
        Objective_DIRECT = np.append(Objective_DIRECT,sol_DIRECT.fun)
        FunCall_DIRECT = np.append(FunCall_DIRECT, sol_DIRECT.nfev)
        
        sol_shgo = optimize.shgo(ackley_function,bounds,minimizer_kwargs={'maxfev':10000})
        Objective_shgo = np.append(Objective_shgo,sol_shgo.fun)
        FunCall_shgo = np.append(FunCall_shgo, sol_shgo.nfev)
        
        sol_DA = optimize.dual_annealing(ackley_function,bounds)
        Objective_DA = np.append(Objective_DA,sol_DA.fun)
        FunCall_DA = np.append(FunCall_DA, sol_DA.nfev)
        
        sol_DE = optimize.differential_evolution(ackley_function,bounds)
        Objective_DE = np.append(Objective_DE,sol_DE.fun)
        FunCall_DE = np.append(FunCall_DE, sol_DE.nfev)
        
        sol_BasinHop = optimize.basinhopping(ackley_function, x0, stepsize=0.5)
        Objective_BasinHop = np.append(Objective_BasinHop,sol_BasinHop.fun)
        FunCall_BasinHop = np.append(FunCall_BasinHop, sol_BasinHop.nfev)
        

# Plot the arrays
plt.figure()
plt.plot(NumDim, FunCall_DIRECT, label="DIRECT", marker='o', linestyle='-', color='b')
plt.plot(NumDim, FunCall_shgo, label="shgo", marker='^', linestyle='-', color='g')
plt.plot(NumDim, FunCall_DA, label="DA", marker='s', linestyle='-', color='r')
plt.plot(NumDim, FunCall_DE, label="DE", marker='*', linestyle='-', color='m')
plt.plot(NumDim, FunCall_BasinHop, label="Basin Hopping", marker='x', linestyle='-', color='c')
plt.yscale('log')  # Set y-axis to logarithmic scale

# Add labels, title, and legend
plt.xlabel("Dimension")
plt.ylabel("Function calls to reach optima")
plt.title("Efficiency Comparison of Global Optimizers in SciPy for Multi-dimensional Ackley Function")
plt.legend()

plt.show()

plt.figure()
plt.plot(NumDim, Objective_DIRECT, label="DIRECT", marker='o', linestyle='-', color='b')
plt.plot(NumDim, Objective_shgo, label="shgo", marker='^', linestyle='-', color='g')
plt.plot(NumDim, Objective_DA, label="DA", marker='s', linestyle='-', color='r')
plt.plot(NumDim, Objective_DE, label="DE", marker='*', linestyle='-', color='m')
plt.plot(NumDim, Objective_BasinHop, label="Basin Hopping", marker='x', linestyle='-', color='c')
plt.yscale('log')  # Set y-axis to logarithmic scale

# Add labels, title, and legend
plt.xlabel("Dimension")
plt.ylabel("Optimal Objective Value")
plt.title("Efficiency Comparison of Global Optimizers in SciPy for Multi-dimensional Ackley Function")
plt.legend()

plt.show()
        