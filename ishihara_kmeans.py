import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    """
    Randomly initialize centroids from the input data.

    Parameters:
        data (ndarray): Flattened image pixels (num_samples, 3).
        k (int): Number of clusters.

    Returns:
        ndarray: Initialized centroids (k, 3).
    """
    np.random.seed(1)
    return data[np.random.choice(len(data), k, replace=False)]

def assign_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid.

    Parameters:
        data (ndarray): Flattened image pixels.
        centroids (ndarray): Current centroids.

    Returns:
        ndarray: Cluster index for each pixel.
    """
    distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k, prev_centroids):
    """
    Recalculate centroids as the mean of all points in each cluster.

    Parameters:
        data (ndarray): Flattened image pixels.
        labels (ndarray): Cluster assignments.
        k (int): Number of clusters.
        prev_centroids (ndarray): Previous centroids.

    Returns:
        ndarray: Updated centroids.
    """
    new_centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(prev_centroids[i])
    return np.array(new_centroids)

def kmeans(data, k, max_iter=100, epsilon=1e-4):
    """
    Perform K-means clustering on image pixel data.

    Parameters:
        data (ndarray): Flattened image pixels.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Convergence threshold.

    Returns:
        labels (ndarray): Cluster assignments.
        centroids (ndarray): Final centroids.
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k, centroids)
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids
    return labels, centroids

def visualize_clusters(image_np, label_map, k):
    """
    Visualize each cluster in green for manual selection.

    Parameters:
        image_np (ndarray): Original image array.
        label_map (ndarray): Cluster map reshaped to image dimensions.
        k (int): Number of clusters.
    """
    print("Inspecting clusters (green = candidate region):")
    rows, cols = 3, (k + 2) // 3
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i in range(k):
        mask = (label_map == i)
        temp_image = np.zeros_like(image_np)
        temp_image[mask] = [0, 255, 0]
        axs[i].imshow(temp_image)
        axs[i].set_title(f"Cluster {i}")
        axs[i].axis("off")

    for j in range(k, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

def highlight_number(image_np, label_map, number_clusters):
    """
    Highlight the clusters identified as the number using green.

    Parameters:
        image_np (ndarray): Original image array.
        label_map (ndarray): Cluster labels per pixel.
        number_clusters (list): Indices of clusters representing the number.

    Returns:
        ndarray: Image with number highlighted in green.
    """
    result_image = np.zeros_like(image_np)
    for cluster_id in number_clusters:
        result_image[label_map == cluster_id] = [0, 255, 0]
    return result_image

def show_final_output(original, extracted):
    """
    Display the original image and the extracted number side by side.

    Parameters:
        original (ndarray): Original image array.
        extracted (ndarray): Highlighted image array.
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Extracted Number (Green)")
    plt.imshow(extracted)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_id = input("Enter image number (without extension): ").strip()
    image_path = f"input_images/{image_id}.jpg"
    k = input("Enter number of clusters (e.g., 3): ").strip()
    k = int(k) if k.isdigit() else 3

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    pixels = image_np.reshape((-1, 3)).astype(np.float32)

    # Apply K-means
    labels, centroids = kmeans(pixels, k)
    label_map = labels.reshape(image_np.shape[:2])

    # Visualize clusters to choose number
    visualize_clusters(image_np, label_map, k)

    # Get cluster selection from user
    selection = input("Enter cluster indices (comma-separated) that represent the number: ")
    number_clusters = [int(x.strip()) for x in selection.split(",")]

    # Highlight and save result
    result_image = highlight_number(image_np, label_map, number_clusters)
    Image.fromarray(result_image).save(f"output_images/{image_id}_number_only.png")

    # Show final comparison
    show_final_output(image_np, result_image)