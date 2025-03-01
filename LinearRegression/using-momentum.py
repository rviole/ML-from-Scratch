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
TARGET_B = 3
epoch = 0

x = np.arange(-10, 10, 0.1)
y = x * TARGET_K + TARGET_B


lr_k = 0.0001  # learning rate
lr_b = 0.001  # learning rate
beta = 0.95  # how much of momentum to keep

v_k = 0
prev_v_k = 0

v_b = 0
prev_v_b = 0

history = {
    "epoch": [],
    "k_velocity": [],
    "k_previous_velocity": [],
    "b_velocity": [],
    "b_previous_velocity": [],
    "k_gradient": [],
    "b_gradient": [],
    "k": [],
    "b": [],
    "loss": [],
    "k_lr": [],
    "b_lr": [],
}

MAX_EPOCHS = 1500
# initial k value
initial_k = -5
initial_b = -5

k = initial_k
b = initial_b


while True:
    y_pred = x * k + b
    error = y - y_pred
    loss = np.mean(error**2)

    if loss < 0.0001 or epoch > MAX_EPOCHS:
        break

    k_gradient = -2 * np.mean(x * error)
    b_gradient = -2 * np.mean(error)

    prev_v_k = v_k
    v_k = (prev_v_k * beta) + (k_gradient * lr_k)
    k = k - v_k

    prev_v_b = v_b
    v_b = (prev_v_b * beta) + (b_gradient * lr_b)
    b = b - v_b

    lr_k = lr_k * 1.001
    lr_b = lr_b * 1.001

    history["epoch"].append(epoch)
    history["k_velocity"].append(float(f"{v_k:.2f}"))
    history["k_previous_velocity"].append(float(f"{prev_v_k:.2f}"))
    history["b_velocity"].append(float(f"{v_b:.2f}"))
    history["b_previous_velocity"].append(float(f"{prev_v_b:.2f}"))
    history["k_gradient"].append(float(f"{k_gradient:.2f}"))
    history["b_gradient"].append(float(f"{b_gradient:.2f}"))
    history["k"].append(float(f"{k:.2f}"))
    history["b"].append(float(f"{b:.2f}"))
    history["loss"].append(float(f"{loss:.2f}"))
    history["k_lr"].append(float(f"{lr_k:.2f}"))
    history["b_lr"].append(float(f"{lr_b:.2f}"))
    epoch += 1


show_history(history)


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
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

# 4th plot
ax[1, 1].plot(history["b"], history["loss"], label="Loss", color="black", lw=2)
ax[1, 1].axvline(x=TARGET_B, lw=3, label=f"Target b: {TARGET_B}", color="blue")
ax[1, 1].plot(
    history["b"][0],
    history["loss"][0],
    marker="o",
    color="green",
    label=f"Start: b={initial_b}",
    markersize=10,
)
ax[1, 1].plot(
    history["b"][-1],
    history["loss"][-1],
    marker="o",
    color="red",
    label=f"End: b={history['b'][-1]}",
    markersize=10,
)
ax[1, 1].legend()
ax[1, 1].grid()
ax[1, 1].set_title("Loss vs b")
ax[1, 1].set_xlabel("b")
ax[1, 1].set_ylabel("Loss")


plt.tight_layout()
plt.show()


# plt.figure(figsize=(12, 10))
# plt.plot(x, y, label="Actual", color="blue", lw=2)
# plt.plot(x, y_pred, label="Predicted", color="red", lw=2)


# plt.plot(history["k"], history["loss"], label="Loss", color="black", lw=2)
# plt.xlabel("k")
# plt.ylabel("Loss")
# plt.title("Loss vs k")

# plt.axhline(y=0, color="red", lw=1)
# plt.axvline(x=TARGET_K, lw=3, label=f"Target k: {TARGET_K}", color="blue")
# plt.plot(history["k"][0], history["loss"][0], marker="o", color="green", label=f"Start: k={initial_k}", markersize=10)
# plt.plot(history["k"][-1], history["loss"][-1], marker="o", color="red", label=f"End: k={history['k'][-1]}", markersize=10)

# plt.legend()
# plt.grid()
# plt.show()
