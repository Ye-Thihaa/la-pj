import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import SpectralClustering
import tkinter as tk
from tkinter import filedialog


def segment_image_spectral(image, num_clusters=5, downscale=150, neighbors=50):
    """
    Segments an image into regions using Spectral Clustering.
    Returns both a label map and a color-segmented image.
    """
    h, w, _ = image.shape
    scale = downscale / max(h, w)
    small_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixel_data = small_image.reshape(-1, 3)

    print(f"Performing spectral clustering with {num_clusters} clusters and {neighbors} neighbors...")

    clustering = SpectralClustering(
        n_clusters=num_clusters,
        assign_labels='kmeans',
        random_state=0,
        affinity='nearest_neighbors',
        n_neighbors=neighbors
    )

    labels = clustering.fit_predict(pixel_data)
    segmented_labels = labels.reshape(small_image.shape[:2])

    # Map each cluster to its average color
    segmented_colored = np.zeros_like(small_image)
    for k in range(num_clusters):
        segmented_colored[segmented_labels == k] = np.mean(pixel_data[labels == k], axis=0)

    return segmented_labels, segmented_colored


if __name__ == "__main__":
    try:
        # Use a Tkinter file dialog to get the image path
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        input_filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if not input_filename:
            print("No file was selected. Exiting.")
            sys.exit(1)

        if not os.path.exists(input_filename):
            print(f"Error: The file '{input_filename}' was not found.")
            sys.exit(1)

        # Read the image using OpenCV
        original_image_bgr = cv2.imread(input_filename)
        if original_image_bgr is None:
            raise ValueError(f"Could not read the image from '{input_filename}'.")

        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

        NUM_CLUSTERS = 6
        segmented_map, segmented_image = segment_image_spectral(
            original_image_rgb,
            num_clusters=NUM_CLUSTERS,
            downscale=150,
            neighbors=50
        )

        print("\n✅ Segmentation complete!")

        # Show results
        plt.figure(figsize=(18, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(segmented_map, cmap='viridis')
        plt.title(f"Label Map ({NUM_CLUSTERS} clusters)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title("Segmented Image (Cluster Colors)")
        plt.axis("off")

        plt.show()

    except Exception as e:
        print(f"\nAn error occurred: {e}")