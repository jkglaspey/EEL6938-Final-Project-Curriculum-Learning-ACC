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

def generate_combined_visual(image_paths, output_path, title, labels):
    images = [crop_whitespace(Image.open(path)) for path in image_paths]
    n_images = len(images)
    n_cols = 1
    n_rows = n_images

    # Get image dimensions (assume all same size)
    img_width, img_height = images[0].size
    scale = 1 / 100  # Convert pixels to inches (for dpi=100)
    fig_w = img_width * scale
    fig_h = img_height * n_rows * scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=100)
    if n_rows == 1:
        axes = [axes]

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(labels[i], fontsize=18, pad=8)
        axes[i].axis("off")

    fig.suptitle(title, fontsize=22, y=0.96)
    plt.subplots_adjust(wspace=0, hspace=0.25, top=0.92)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# === Configuration ===

base_dir = "./logs_chunk_training/"
output_dir = "./combined_visuals"
os.makedirs(output_dir, exist_ok=True)

schedules = {
    "schedule-curriculum": "Curriculum",
    "schedule-anti": "Anti-Curriculum",
    "schedule-baseline": "Baseline"
}

image_filenames = [
    "speed_comparison.png",
    "following_distance.png",
    "jerk_analysis.png",
    "reward_penalty_over_time.png",
    "speed_difference.png"
]

image_titles = {
    "speed_comparison.png": "Speed Comparison Across Training Schedules",
    "following_distance.png": "Following Distance Across Training Schedules",
    "jerk_analysis.png": "Jerk Analysis Across Training Schedules",
    "reward_penalty_over_time.png": "Reward Over Time During Training",
    "speed_difference.png": "Speed Difference: Ego vs Lead Vehicle"
}

# === Run Loop ===

for image_name in image_filenames:
    image_paths = []
    labels = []

    for schedule_folder, label in schedules.items():
        img_path = os.path.join(base_dir, schedule_folder, image_name)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)

    if not image_paths:
        print(f"[WARNING] Skipping {image_name} — no images found.")
        continue

    output_path = os.path.join(output_dir, f"combined_{image_name}")
    generate_combined_visual(
        image_paths=image_paths,
        output_path=output_path,
        title=image_titles.get(image_name, image_name),
        labels=labels
    )