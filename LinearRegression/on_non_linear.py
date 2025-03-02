import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def show_history(history: dict):
    """
    Display the training history in a tabular format.

    Args:
        history (dict): A dictionary containing the training history with keys as column names and values as lists of data points.
    """
    # Create a PrettyTable object to display the history
    table = PrettyTable()

    # Add each key and its corresponding values as a column in the table
    for key in history.keys():
        table.add_column(key, history[key])

    # Print the table
    print(table)


TARGET_K = 2
epoch = 0

x = np.arange(-10, 10, 0.1)
y = x**2 * TARGET_K


derivative = 4 * x

# plt.plot(x, y)
# plt.plot(x, derivative)
# plt.show()
# exit(1)


class MomentumOptimizer:
    def __init__(self, lr=0.0001, beta=0.9):
        # WARNING - LR in this implementation is constant
        self.lr = lr
        self.beta = beta
        self.momentum = 0
        self.previous_momentum = 0

    def step(self, gradient):
        self.previous_momentum = self.momentum
        self.momentum = (self.previous_momentum * self.beta) + (gradient * self.lr)

        return self.momentum


lr_k = 0.000005  # learning rate

k_optimizer = MomentumOptimizer(lr=lr_k)

history = {
    "epoch": [],
    "k_velocity": [],
    "k_previous_velocity": [],
    "k_gradient": [],
    "k": [],
    "loss": [],
    "k_lr": [],
}

MAX_EPOCHS = 1500
# initial k value
initial_k = -5
k = initial_k


while True:
    y_pred = x**2 * k
    error = y - y_pred
    loss = np.mean(error**2)

    if loss < 0.0001 or epoch > MAX_EPOCHS:
        break

    k_gradient = -2 * np.mean(x**2 * error)
    k = k - k_optimizer.step(k_gradient)
    lr_k = lr_k * 1.001

    history["epoch"].append(epoch)
    history["k_velocity"].append(float(f"{k_optimizer.momentum:.2f}"))
    history["k_previous_velocity"].append(float(f"{k_optimizer.previous_momentum:.2f}"))
    history["k_gradient"].append(float(f"{k_gradient:.2f}"))
    history["k"].append(float(f"{k:.2f}"))
    history["loss"].append(float(f"{loss:.2f}"))
    history["k_lr"].append(float(f"{lr_k:.2f}"))

    epoch += 1


show_history(history)


fig, ax = plt.subplots(2, 2, figsize=(12, 7))
# 1st plot - real vs predicted
# 2nd plot - loss vs epoch
# 3rd plot - k vs loss
# 4th plot - b vs loss

# 1st plot
ax[0, 0].plot(x, y, label="Actual", color="blue", lw=7)
ax[0, 0].plot(x, y_pred, label="Predicted", color="red", lw=2)
ax[0, 0].legend()
ax[0, 0].set_title("Actual vs Predicted")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")

# 2nd plot
ax[0, 1].plot(history["epoch"], history["loss"], label="Loss", color="black", lw=2)
ax[0, 1].axhline(y=0, color="red", lw=1)
ax[0, 1].legend()
ax[0, 1].set_title("Loss vs Epoch")
ax[0, 1].set_xlabel("Epoch")
ax[0, 1].set_ylabel("Loss")

# 3rd plot
ax[1, 0].plot(history["k"], history["loss"], label="Loss", color="black", lw=2)
ax[1, 0].axvline(x=TARGET_K, lw=3, label=f"Target k: {TARGET_K}", color="blue")
ax[1, 0].plot(
    history["k"][0],
    history["loss"][0],
    marker="o",
    color="green",
    label=f"Start: k={initial_k}",
    markersize=10,
)
ax[1, 0].plot(
    history["k"][-1],
    history["loss"][-1],
    marker="o",
    color="red",
    label=f"End: k={history['k'][-1]}",
    markersize=10,
)
ax[1, 0].legend()
ax[1, 0].grid()
ax[1, 0].set_title("Loss vs k")
ax[1, 0].set_xlabel("k")
ax[1, 0].set_ylabel("Loss")


plt.tight_layout()
plt.show()
