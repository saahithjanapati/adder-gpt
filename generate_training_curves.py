import pickle
import matplotlib.pyplot as plt

# Load the training log
with open('training_log.pkl', 'rb') as f:
    training_log = pickle.load(f)

# Extract data
num_iter = training_log['num_iter']
training_loss = training_log['training_loss']
training_acc = training_log['training_acc']
val_loss = training_log['val_loss']
val_acc = training_log['val_acc']

# Create a plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Training and Validation Loss on the left y-axis
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(num_iter, training_loss, label='Training Loss', marker='o', linestyle='-', color='tab:blue')
ax1.plot(num_iter, val_loss, label='Validation Loss', marker='o', linestyle='--', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
ax1.grid()

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.plot(num_iter, training_acc, label='Training Accuracy', marker='o', linestyle='-', color='tab:orange')
ax2.plot(num_iter, val_acc, label='Validation Accuracy', marker='o', linestyle='--', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper right')

# Add a title and save the figure
plt.title('Training and Validation Loss & Accuracy')
plt.savefig('training_validation_combined.png')
plt.close()