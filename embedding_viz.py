import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Directory containing the embedding tensors
embedding_dir = "embeddings/"
output_gif = "embedding_evolution.gif"
final_pca_image = "final_embeddings_pca.png"

# List all .pt files and sort them numerically
embedding_files = sorted(
    [f for f in os.listdir(embedding_dir) if f.endswith(".pt")],
    key=lambda x: int(x.split('.')[0])
)
print(f"Found {len(embedding_files)} embedding files.")

# Load the embeddings from the last checkpoint (14000.pt) onto the CPU
print("Loading final embeddings (14000.pt)...")
final_embeddings = torch.load(os.path.join(embedding_dir, "14000.pt"), map_location=torch.device('cpu'))
final_embeddings = final_embeddings.detach().numpy()  # Detach from computation graph
print("Final embeddings loaded.")

# Perform PCA on the embeddings from the last checkpoint
print("Performing PCA on the final embeddings...")
pca = PCA(n_components=2)
final_transformed = pca.fit_transform(final_embeddings)
print("PCA transformation fitted.")

# Save a static plot of the PCA-transformed final embeddings
print("Saving PCA plot for final embeddings...")
plt.figure(figsize=(6, 6))
plt.scatter(final_transformed[:, 0], final_transformed[:, 1], c=range(10), cmap="viridis", s=100)
plt.title("Final Embeddings (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Digit")
plt.grid()
plt.savefig(final_pca_image)
plt.close()  # Close the figure to free memory
print(f"Final PCA image saved as {final_pca_image}")

# Apply PCA transformation to all embeddings and store the results
print("Applying PCA transformation to all embedding files...")
all_transformed_embeddings = []
for idx, file in enumerate(embedding_files):
    embeddings = torch.load(os.path.join(embedding_dir, file), map_location=torch.device('cpu'))
    embeddings = embeddings.detach().numpy()  # Detach from computation graph
    transformed = pca.transform(embeddings)
    step = int(file.split('.')[0])
    all_transformed_embeddings.append((step, transformed))
    if idx % 5 == 0 or idx == len(embedding_files) - 1:  # Progress message every 5 files
        print(f"Processed {idx + 1}/{len(embedding_files)} files.")

# Plotting setup for GIF creation
print("Creating GIF animation...")
fig, ax = plt.subplots(figsize=(6, 6))

# Initialize with the first frame's data
initial_step, initial_transformed = all_transformed_embeddings[0]
scatter = ax.scatter(
    initial_transformed[:, 0],
    initial_transformed[:, 1],
    c=range(10),  # Assuming 10 distinct classes/digits
    cmap="viridis",
    s=100
)

# Add number labels for each point
texts = [
    ax.text(x, y, str(i), fontsize=8, ha='right', va='bottom')
    for i, (x, y) in enumerate(initial_transformed)
]

# Set plot limits based on the final transformed embeddings
ax.set_xlim(final_transformed[:, 0].min() - 1, final_transformed[:, 0].max() + 1)
ax.set_ylim(final_transformed[:, 1].min() - 1, final_transformed[:, 1].max() + 1)
ax.set_title(f"Embedding Evolution - Step {initial_step}")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

# Optional: Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Digit")

def update(frame):
    step, transformed = all_transformed_embeddings[frame]
    scatter.set_offsets(transformed)
    ax.set_title(f"Embedding Evolution - Step {step}")
    for i, (x, y) in enumerate(transformed):
        texts[i].set_position((x, y + 0.2))  # Adjust the offset as needed
    return (scatter,) + tuple(texts)

# Create animation
anim = FuncAnimation(fig, update, frames=len(all_transformed_embeddings), interval=200, blit=True)

# Save animation as GIF
anim.save(output_gif, writer="imagemagick")
plt.close()  # Close the figure to free memory
print(f"GIF saved as {output_gif}")