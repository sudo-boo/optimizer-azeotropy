import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def blackbox_function(x):
    if len(x.shape) == 1:
        return (4 - 2.1*x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2
    else:
        return (4 - 2.1*x[:, 0]**2 + (x[:, 0]**4)/3) * x[:, 0]**2 + x[:, 0]*x[:, 1] + (-4 + 4*x[:, 1]**2) * x[:, 1]**2

def acquisition_function(x, gp, y_opt, xi=0.01):
    mu, sigma = gp.predict(x, return_std=True)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide='warn'):
        imp = mu - y_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return -ei

def optimize_bayesian_optimization(bounds, max_iter=15):
    x_list = []
    y_list = []
    
    # Initialize Gaussian Process
    kernel = RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # Initial random point
    x_initial = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
    y_initial = blackbox_function(x_initial)
    x_list.append(x_initial)
    y_list.append(y_initial)
    
    for i in range(max_iter):
        x_train = np.concatenate(x_list, axis=0)
        y_train = np.array(y_list)
        
        # Fit Gaussian Process
        gp.fit(x_train, y_train)
        
        # Minimize acquisition function
        acquisition_func = lambda x: acquisition_function(x.reshape(-1, bounds.shape[0]), gp, np.min(y_train))
        x_min = minimize(acquisition_func, bounds=bounds, method='L-BFGS-B').x
        x_list.append(x_min)
        
        # Evaluate the black-box function
        y_min = blackbox_function(x_min)
        y_list.append(y_min)
        
    return x_list, y_list

# Define the search space bounds
bounds = np.array([[200, 300], [25, 75]])

# Optimize using Bayesian Optimization
x_optimal, y_optimal = optimize_bayesian_optimization(bounds)

# Plot the function
x = np.linspace(200, 300, 100)
y = np.linspace(25, 75, 100)
X, Y = np.meshgrid(x, y)
Z = blackbox_function(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Function Value')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')

# Plot the points evaluated during optimization
x_points = np.array(x_optimal)[:, 0]
y_points = np.array(x_optimal)[:, 1]
z_points = y_optimal
plt.scatter(x_points, y_points, c=z_points, cmap='hot', label='Optimization Points')
plt.colorbar(label='Function Value')
plt.legend()
plt.title('Black-box Function and Optimization Progress')
plt.show()
