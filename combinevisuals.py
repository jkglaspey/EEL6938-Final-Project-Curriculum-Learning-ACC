import os
import math
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

def crop_whitespace(img, bg_color=(255, 255, 255), tolerance=5):
    """
    Crops white (or near-white) space from the edges of an image.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    bg = Image.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    
    if bbox:
        return img.crop(bbox)
    else:
        return img  # return original if no content found


# Set your parent directory
base_dir = "./logs_chunk_training/Tau/"

# List of image filenames to combine
image_filenames = [
    "following_distance.png",
    "jerk_analysis.png",
    "reward_penalty_over_time.png",
    "speed_comparison.png",
    "speed_difference.png"
]

# Optional display titles for each type
image_titles = {
    "following_distance.png": "Following Distance Analysis",
    "jerk_analysis.png": "Jerk Analysis",
    "reward_penalty_over_time.png": "Reward Over Time",
    "speed_comparison.png": "Speed Comparison",
    "speed_difference.png": "Speed Difference with Lead Vehicle"
}

# Provide custom labels for each folder (instead of folder name)
custom_labels = {
    "model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.01_gamma-0.99_ent-auto_arch-256-256": "Tau: 0.01",
    "model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.001_gamma-0.99_ent-auto_arch-256-256": "Tau: 0.001",
    "model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.0001_gamma-0.99_ent-auto_arch-256-256": "Tau: 0.0001",
    "model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.02_gamma-0.99_ent-auto_arch-256-256": "Tau: 0.02",
    "model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.005_gamma-0.99_ent-auto_arch-256-256": "Tau: 0.005",
}

# Output directory
output_dir = "combined_visuals"
os.makedirs(output_dir, exist_ok=True)

# Get sorted list of child folders
child_folders = sorted([
    f for f in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, f))
])

for image_name in image_filenames:
    images = []
    labels = []

    for folder in child_folders:
        img_path = os.path.join(base_dir, folder, image_name)
        if os.path.exists(img_path):
            original_img = Image.open(img_path)
            cropped_img = crop_whitespace(original_img)
            images.append(cropped_img)
            label = custom_labels.get(folder, folder)
            labels.append(label)

    if not images:
        print(f"No images found for {image_name}, skipping.")
        continue

    n_images = len(images)
    n_cols = 2
    n_rows = math.ceil(n_images / n_cols)

    # Get image dimensions (assume all same size)
    img_width, img_height = images[0].size
    scale = 1 / 100  # Scale down from pixels to inches (DPI = 100)
    fig_w = img_width * n_cols * scale
    fig_h = img_height * n_rows * scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=100)
    axes = axes.flatten()

    for ax in axes[n_images:]:
        ax.axis("off")

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(labels[i], fontsize=14, pad=2)
        axes[i].axis("off")

    fig.suptitle(image_titles.get(image_name, image_name), fontsize=14)
    plt.subplots_adjust(wspace=0, hspace=0, top=0.95)

    output_path = os.path.join(output_dir, f"combined_{image_name}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
