import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize

# Define your black-box function (This was just to test the code)
# def black_box_function(x, y):
#     return np.sin(x + y**2)

# def plot_black_box_function():
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     x_range = np.linspace(bounds[0][0], bounds[0][1], 100)
#     y_range = np.linspace(bounds[1][0], bounds[1][1], 100)
#     X, Y = np.meshgrid(x_range, y_range)
#     Z = black_box_function(X, Y)
#     ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.9)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Output')
#     ax.set_title('Black Box Function')
#     plt.show()


# --------------------------------------------------------------------------------------------

# Define the bounds in our case C and R
bounds = [(200, 300), (25, 75)]
# bounds = [(-1, 1), (-1, 1)] # This was just for testing the above function

# --------------------------------------------------------------------------------------------

X_init = np.array([[250, 50], [263.5, 75], [231, 75], [244.3812, 25], [235.7397, 75], [300, 25], [300, 75], [260, 35], [230, 50], [205, 40], [240, 45], [240, 60], [275, 35], [240, 51], [210, 68], [225, 42], [260, 42]])

# --------------------------------------------------------------------------------------------

# y_init = black_box_function([X_init[0][0]], [X_init[0][1]]) (This was just for testing the above function)
y_init = np.array([34.9326, 223.5943, 176.11, 143.2747, 181, 107.794, 311.9724, 57.69, 36.7465, 89.3282, 36.5198, 63.5974, 55.3876, 35.1882, 102.1441, 53.8587, 36.3241])

# --------------------------------------------------------------------------------------------

# Define the Gaussian Process regressor with an RBF kernel
kernel = C(1.0, (1e-1, 1e3)) * RBF(1.0, (1e-1, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# --------------------------------------------------------------------------------------------

# Fit the Gaussian Process model
gp.fit(X_init, y_init)

# --------------------------------------------------------------------------------------------

# Define the acquisition function (expected improvement)
# This is the function that will be minimized to find the next best point

def acquisition_function(x):
    x = np.atleast_2d(x)  # Ensure x is 2D array
    mean, std = gp.predict(x, return_std=True)
    best_observed = np.min(y_init)
    gamma = (best_observed - mean) / std
    expected_improvement = std * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    return -expected_improvement 

# --------------------------------------------------------------------------------------------

# Function to update the GP model with new data points
# This is useful when we have evaluated the black-box function at the next best point

def update_gp(new_X, new_y):
    global X_init, y_init, gp
    X_init = np.vstack((X_init, new_X))
    y_init = np.concatenate((y_init, new_y))
    gp.fit(X_init, y_init)

# --------------------------------------------------------------------------------------------

# Basically plotting stuff
    
def plot_functions():
    x_range = np.linspace(bounds[0][0], bounds[0][1], 50)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x_range, y_range)    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the predicted function
    Z_pred = gp.predict(np.vstack((X.ravel(), Y.ravel())).T).reshape(X.shape)
    ax.plot_surface(X, Y, Z_pred, cmap='inferno', alpha=0.9, label='Predicted Function')
    
    # Plot the data points
    ax.scatter(X_init[:, 0], X_init[:, 1], y_init, color='red', s=100, label='Data Points')
    
    # Compute and plot the 95% confidence interval
    mean, std = gp.predict(np.vstack((X.ravel(), Y.ravel())).T, return_std=True)
    lower_bound = mean - 1.96 * std  # 95% confidence interval
    upper_bound = mean + 1.96 * std
    lower_bound = lower_bound.reshape(X.shape)
    upper_bound = upper_bound.reshape(X.shape)
    ax.plot_surface(X, Y, lower_bound, color='blue', alpha=0.15)
    ax.plot_surface(X, Y, upper_bound, color='blue', alpha=0.15)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Output')
    ax.set_title('Gaussian Process Regression')
    ax.legend()
    
    plt.show()

# --------------------------------------------------------------------------------------------



# << Find the next best point and evaluate the black-box function (IN OUR CASE, THE OPTIMIZER DATA POINTS) >>
# --------------------------------------------------------------------------------------------
# This is the main loop of the Bayesian optimization algorithm
# This is commented as it was used during the training of the model
# --------------------------------------------------------------------------------------------
    
# def find_next_best_point():
#     # Define the bounds for the optimization (same as the variable bounds)
#     bounds_opt = [(bound[0], bound[1]) for bound in bounds]
#     # Find the minimum of the acquisition function
#     result = minimize(acquisition_function, x0=np.random.uniform(bounds[0][0], bounds[0][1], 2),
#                       bounds=bounds_opt)
#     # Extract the next best point
#     next_best_point = result.x    
#     return next_best_point


# for i in range(16):
#     next_best_point = find_next_best_point()
#     print("Next best point:", next_best_point)
#     # Evaluate the function at the next best point
#     next_best_value = float(input("Value at next best point:"))
#     # next_best_value = black_box_function(*next_best_point)
#     print("Value at next best point:", next_best_value)
#     with open('data.txt', 'a') as f:
#         f.write(str(next_best_point) + "\n")
#         f.write(str(next_best_value) + "\n")
#     # Update the GP model with the new data point
#     update_gp(np.atleast_2d(next_best_point), np.atleast_1d(next_best_value))
# 
# --------------------------------------------------------------------------------------------

plot_functions()
# plot_black_box_function()

# --------------------------------------------------------------------------------------------

# Predict the minimum value and corresponding variables
# Here, we use the GP model to predict the minimum value and the corresponding variables
# The values are calculated based upon the mean of the GP model that we have trained
# The minimum value is the minimum of the mean values for the grid points
# The corresponding variables are the variables for the grid point with the minimum mean value

def predict_minimum():
    # Generate a grid of points within the bounds
    x_range = np.linspace(bounds[0][0], bounds[0][1], 200)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    # Predict the mean and standard deviation values for the grid points
    mean, std = gp.predict(points, return_std=True)
    min_index = np.argmin(mean)

    # Get the corresponding variables
    min_x, min_y = points[min_index]

    # Calculate the minimum value and its variance
    min_value = mean[min_index]
    min_variance = std[min_index] ** 2  # Variance is the square of standard deviation

    return min_x, min_y, min_value, min_variance

# Find the minimum value, corresponding variables, and variance
min_x, min_y, min_value, min_variance = predict_minimum()

# --------------------------------------------------------------------------------------------


# Print the result
print("Minimum Value:", min_value)
print("Variance at Minimum Point:", min_variance)
print("Corresponding Variables (X, Y):", min_x, min_y)

# --------------------------------------------------------------------------------------------