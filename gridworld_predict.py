
"""Run gridworld with different reward functions
   Interpolate value functions for new reward functions
   It becomes a supervised learning problem

   Weights for terminal rewards are suppressed with flag
   """
import numpy as np
from gridworld import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import matplotlib.pyplot as plt
living_reward_interpolate_flag = True
terminal_interpolate_flag = False
if living_reward_interpolate_flag:
    living_reward_weight = np.arange(0, -0.5, -0.1)
    print("Living reward weight = ", living_reward_weight)
if terminal_interpolate_flag:
    terminal_weight = np.arange(1, 5, 0.5)
    print("Terminal weight = ", terminal_weight)
wind = 0.1
# contains the state and reward pair
X = []
# contains the value at the given state
y = []
if living_reward_interpolate_flag:
    for i in living_reward_weight:
        grid = GridWorld(wind=wind, living_reward=i, positive_scalar_factor=1,
                         negative_scalar_factor=1)
        grid.run_value_iteration()
        reward = grid.reward[1:4, 1:5]
        value = grid.state_value_function[1:4, 1:5]

        for j in range(reward.shape[0]):
            for k in range(reward.shape[1]):
                X.append([j, k, reward[j, k]])
                y.append(value[j, k])
if terminal_interpolate_flag:
    for i in terminal_weight:
        grid = GridWorld(wind=wind, living_reward=-0.02, positive_scalar_factor=i,
                         negative_scalar_factor=1)
        grid.run_value_iteration()
        reward = grid.reward[1:4, 1:5]
        value = grid.state_value_function[1:4, 1:5]

        # Need to normalize for rewards greater than 1
        # Gaussian process wants standardized values
        reward /= np.max(np.abs(reward))
        value /= np.max(np.abs(value))
        for j in range(reward.shape[0]):
            for k in range(reward.shape[1]):
                X.append([j, k, reward[j, k]])
                y.append(value[j, k])
X = np.array(X)
y = np.array(y)

"""Run Gaussian process"""

# Instantiate a Gaussian Process model
# kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Default evaluation points
eval_reward = -0.02
eval_terminal_weight = 1
# Example evaluation points
if living_reward_interpolate_flag:
    eval_reward = -0.23

    grid = GridWorld(wind=wind, living_reward=eval_reward,
                     positive_scalar_factor=eval_terminal_weight, negative_scalar_factor=1)
    grid.run_value_iteration()
    reward = grid.reward[1:4, 1:5]
    value = grid.state_value_function[1:4, 1:5]
    reward_max = np.max(np.abs(reward))
    value_max = np.max(np.abs(value))

    X_eval = []
    y_eval = []
    for j in range(reward.shape[0]):
        for k in range(reward.shape[1]):
            X_eval.append([j, k, reward[j, k]])
            y_eval.append(value[j, k])
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
if terminal_interpolate_flag:
    eval_terminal_weight = 2.2

    grid = GridWorld(wind=wind, living_reward=eval_reward,
                     positive_scalar_factor=eval_terminal_weight, negative_scalar_factor=1)
    grid.run_value_iteration()
    reward = grid.reward[1:4, 1:5]
    value = grid.state_value_function[1:4, 1:5]
    reward_max = np.max(np.abs(reward))
    value_max = np.max(np.abs(value))
    reward /= reward_max
    value /= value_max
    X_eval = []
    y_eval = []
    for j in range(reward.shape[0]):
        for k in range(reward.shape[1]):
            X_eval.append([j, k, reward[j, k]])
            y_eval.append(value[j, k])
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X_eval, return_std=True)
mse = np.mean((y_eval - y_pred)**2)
# y_pred *= value_max
# value *= value_max

value_pred = np.zeros(grid.size)
for i in range(X_eval.shape[0]):
    value_pred[int(X_eval[i, 0]), int(X_eval[i, 1])] = y_pred[i]

sigma_pred = np.zeros(grid.size)
for i in range(X_eval.shape[0]):
    sigma_pred[int(X_eval[i, 0]), int(X_eval[i, 1])] = sigma[i]

np.set_printoptions(precision=6, suppress=True)
print("GAUSSIAN PROCESS = ")
print("value_eval = \n", value)
print("value_pred = \n", value_pred)
print("Mean squared error = ", mse)
print("Sigma given by GP = ", sigma)
print("Median = ", np.median(sigma))

np.set_printoptions(precision=2)
fig1 = plt.figure(1, figsize=(16, 6))
ax1 = fig1.add_subplot(121)
im1 = ax1.imshow(value_pred, cmap='rainbow')
for (j, i), label in np.ndenumerate(value_pred):
    label_text = np.format_float_positional(label, precision=3)
    ax1.text(i, j, label_text, ha='center', va='center', fontsize=28)
ax1.set_title('Predicted values')
ax2 = fig1.add_subplot(122)
im2 = ax2.imshow(value,  cmap='rainbow')
for (j, i), label in np.ndenumerate(value):
    label_text = np.format_float_positional(label, precision=3)
    ax2.text(i, j, label_text, ha='center', va='center', fontsize=28)
ax2.set_title('Actual values')
fig1.subplots_adjust(right=0.8)
cbar_ax = fig1.add_axes([0.85, 0.1, 0.02, 0.7])
fig1.colorbar(im1, cax=cbar_ax)
#plt.title('Gaussian Process prediction with living reward = '+str(eval_reward),
#          x=-20, y=1, fontdict = {'fontsize' : 15})
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
im2 = ax2.imshow(sigma_pred)
for (j, i), label in np.ndenumerate(sigma_pred):
    label_text = np.format_float_positional(label, precision=3)
    ax2.text(i, j, label_text, ha='center', va='center', fontsize=13)
plt.title('Sigma of predicted values', fontdict = {'fontsize' : 15})
plt.show()


