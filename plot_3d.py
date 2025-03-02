import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x1, x2) and its gradient
def f(x1, x2):
    return x1**2 + x2**2 + 5 * np.sin(x1) * np.cos(x2)

def grad_f(x1, x2):
    # Gradient of the function with respect to x1 and x2
    grad_x1 = 2 * x1 + 5 * np.cos(x1) * np.cos(x2)
    grad_x2 = 2 * x2 - 5 * np.sin(x1) * np.sin(x2)
    return grad_x1, grad_x2

# Gradient Descent Parameters
learning_rate = 0.1
epochs = 100
initial_x1, initial_x2 = 5, 5  # Starting point

# Initialize variables for Gradient Descent
x1, x2 = initial_x1, initial_x2
history = [(x1, x2, f(x1, x2))]

# Gradient Descent loop
for epoch in range(epochs):
    grad_x1, grad_x2 = grad_f(x1, x2)
    x1 -= learning_rate * grad_x1
    x2 -= learning_rate * grad_x2
    history.append((x1, x2, f(x1, x2)))

# Extract the history for plotting
history = np.array(history)
x1_history, x2_history, loss_history = history[:, 0], history[:, 1], history[:, 2]

# Create a meshgrid for the surface plot
x1_vals = np.linspace(-5, 5, 100)
x2_vals = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

# Plot the surface and the gradient descent path
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)

# Plot gradient descent path
ax.plot(x1_history, x2_history, loss_history, color='r', marker='o', markersize=5, label='Gradient Descent Path')

ax.set_title("Gradient Descent on f(x1, x2)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
ax.legend()

plt.show()
