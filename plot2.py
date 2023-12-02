import matplotlib.pyplot as plt

# Function to parse the training log text file
def parse_training_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    current_epoch = None

    for line in lines:
        if line.startswith("===Epoch"):
            # Extract the epoch number
            current_epoch = int(line.split()[-1].strip('='))
            epochs.append(current_epoch)
        elif "Training loss:" in line:
            training_loss = float(line.split(": ")[1])
            training_losses.append(training_loss)
        elif "Training accuracy:" in line:
            training_accuracy = float(line.split(": ")[1])
            training_accuracies.append(training_accuracy)
        elif "Validation loss:" in line:
            validation_loss = float(line.split(": ")[1])
            validation_losses.append(validation_loss)
        elif "Validation accuracy:" in line:
            validation_accuracy = float(line.split(": ")[1])
            validation_accuracies.append(validation_accuracy)

    return epochs, training_losses, training_accuracies, validation_losses, validation_accuracies

# Path to your training log text file
log_file_path = 'training_log.txt'

# Parse the training log
epochs, training_losses, training_accuracies, validation_losses, validation_accuracies = parse_training_log(log_file_path)

# Create plots
plt.figure(figsize=(12, 6))

# Training Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_losses, 'o-', label='Training Loss')
plt.plot(epochs, validation_losses, 'o-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracies, 'o-', label='Training Accuracy')
plt.plot(epochs, validation_accuracies, 'o-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
