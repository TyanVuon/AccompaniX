import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend (or another backend of your choice)
import matplotlib.pyplot as plt


# Define the training and validation metrics for each model
models = [
    {"name": "Model 0", "train_loss": [0.2825, 0.1383, 0.1119, 0.0968, 0.0859],
     "val_loss": [0.1684, 0.1414, 0.1270, 0.1243, 0.1197],
     "train_accuracy": [91.85, 95.46, 96.28, 96.74, 97.08],
     "val_accuracy": [94.51, 95.38, 95.87, 96.13, 96.40]},

    {"name": "Model 1", "train_loss": [0.4067, 0.1890, 0.1500, 0.1289, 0.1154],
     "val_loss": [0.2290, 0.1744, 0.1600, 0.1573, 0.1594],
     "train_accuracy": [88.67, 93.91, 95.11, 95.74, 96.14],
     "val_accuracy": [92.71, 94.42, 94.96, 95.20, 95.21]},

    {"name": "Model 2", "train_loss": [0.4001, 0.1998, 0.1605, 0.1394, 0.1247],
     "val_loss": [0.2403, 0.1929, 0.1780, 0.1667, 0.1619],
     "train_accuracy": [88.60, 93.66, 94.76, 95.40, 95.86],
     "val_accuracy": [92.70, 94.16, 94.59, 94.97, 95.26]},

    {"name": "Model 3", "train_loss": [0.4150, 0.1748, 0.1356, 0.1150, 0.1010],
     "val_loss": [0.2332, 0.1797, 0.1587, 0.1538, 0.1494],
     "train_accuracy": [88.26, 94.32, 95.56, 96.16, 96.62],
     "val_accuracy": [92.54, 94.46, 95.11, 95.38, 95.57]},
]

# Create subplots for different metrics
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Plot training and validation loss for each model
for i, model in enumerate(models):
    ax = axes[0, 0]
    ax.plot(range(5), model["train_loss"], label=model["name"])
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(range(5), model["val_loss"], label=model["name"])
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(range(5), model["train_accuracy"], label=model["name"])
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(range(5), model["val_accuracy"], label=model["name"])
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
