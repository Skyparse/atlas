import numpy as np
from atlas.utils.display_utils import plot_imgs

# This Code is Terrible do not use it - purely a quick visualisation script

# Load the data
xA = np.load("./data/02_model_input/mineops_cd_xA.npy")
xB = np.load("./data/02_model_input/mineops_cd_xB.npy")
y_val = np.load("./data/02_model_input/mineops_cd_y.npy")

# Define the scaling factors for R, G, B
r, g, b = 2.9, 2.6, 2.3

# Create RGB images for xA (before)
before_arrays = np.stack(
    [
        xA[:, 2, :, :] * r,  # Red channel
        xA[:, 1, :, :] * g,  # Green channel
        xA[:, 0, :, :] * b,  # Blue channel
    ],
    axis=1,
)

# Create RGB images for xB (after)
after_arrays = np.stack(
    [
        xB[:, 2, :, :] * r,  # Red channel
        xB[:, 1, :, :] * g,  # Green channel
        xB[:, 0, :, :] * b,  # Blue channel
    ],
    axis=1,
)

print("Original shapes:")
print(f"xA shape: {xA.shape}")
print(f"xB shape: {xB.shape}")
print(f"y_val shape: {y_val.shape}")

xA_before = np.transpose(before_arrays, (0, 2, 3, 1))
xB_after = np.transpose(after_arrays, (0, 2, 3, 1))
y_val_reshape = np.transpose(y_val, (0, 2, 3, 1))

print("\nModified shapes:")
print(f"before_arrays shape: {before_arrays.shape}")
print(f"after_arrays shape: {after_arrays.shape}")
print(f"y_val shape: {y_val_reshape.shape}")

# Plot the images using the RGB versions
plot_imgs(
    before_imgs=xA_before,
    after_imgs=xB_after,
    mask_imgs=y_val_reshape,
    nm_img_to_plot=5,
    figsize=3,
)
