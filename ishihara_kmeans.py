import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    """
    Randomly initialize centroids from the input data.

    Parameters:
        data (ndarray): Flattened image pixels of shape (num_samples, 3).
        k (int): Number of clusters.

    Returns:
        ndarray: Initialized centroids of shape (k, 3).
    """
    np.random.seed(1)
    return data[np.random.choice(len(data), k, replace=False)]  # (k, 3)

def assign_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid based on Euclidean distance.

    Parameters:
        data (ndarray): Flattened image pixels of shape (num_samples, 3).
        centroids (ndarray): Current centroids of shape (k, 3).

    Returns:
        ndarray: Cluster index for each pixel, shape (num_samples,)
    """
    # (num_samples, 1, 3) - (1, k, 3) => (num_samples, k, 3)
    distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)  # (num_samples, k)
    return np.argmin(distances, axis=1)  # (num_samples,)

def update_centroids(data, labels, k, prev_centroids):
    """
    Recalculate centroids as the mean of all points in each cluster.

    Parameters:
        data (ndarray): Flattened image pixels of shape (num_samples, 3).
        labels (ndarray): Cluster assignments of shape (num_samples,).
        k (int): Number of clusters.
        prev_centroids (ndarray): Previous centroids of shape (k, 3).

    Returns:
        ndarray: Updated centroids of shape (k, 3).
    """
    new_centroids = []
    for i in range(k):
        cluster_points = data[labels == i]  # (num_points_in_cluster, 3)
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))  # (3,)
        else:
            new_centroids.append(prev_centroids[i])  # fallback
    return np.array(new_centroids)  # (k, 3)

def kmeans(data, k, max_iter=100, epsilon=1e-4):
    """
    Perform K-means clustering on image pixel data.

    Parameters:
        data (ndarray): Flattened image pixels of shape (num_samples, 3).
        k (int): Number of clusters.
        max_iter (int): Max number of iterations.
        epsilon (float): Convergence threshold.

    Returns:
        labels (ndarray): Cluster assignment for each pixel, shape (num_samples,)
        centroids (ndarray): Final centroids of shape (k, 3).
    """
    centroids = initialize_centroids(data, k)  # (k, 3)
    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)  # (num_samples,)
        new_centroids = update_centroids(data, labels, k, centroids)  # (k, 3)
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids
    return labels, centroids

def visualize_clusters(image_np, label_map, k):
    """
    Visualize each cluster in green.

    Parameters:
        image_np (ndarray): Original image array of shape (H, W, 3).
        label_map (ndarray): Cluster map of shape (H, W), each value ∈ [0, k-1].
        k (int): Number of clusters.
    """
    print("Inspecting clusters (green = candidate region):")
    rows, cols = 3, (k + 2) // 3
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i in range(k):
        mask = (label_map == i)  # (H, W), boolean mask
        temp_image = np.zeros_like(image_np)  # (H, W, 3)
        temp_image[mask] = [0, 255, 0]  # highlight cluster i in green
        axs[i].imshow(temp_image)
        axs[i].set_title(f"Cluster {i}")
        axs[i].axis("off")

    for j in range(k, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

def highlight_number(image_np, label_map, number_clusters):
    """
    Highlight selected number clusters in green.

    Parameters:
        image_np (ndarray): Original image array of shape (H, W, 3).
        label_map (ndarray): Cluster labels per pixel, shape (H, W).
        number_clusters (list): Indices of clusters representing the number.

    Returns:
        ndarray: New image with number highlighted in green, shape (H, W, 3).
    """
    result_image = np.zeros_like(image_np)  # (H, W, 3)
    for cluster_id in number_clusters:
        result_image[label_map == cluster_id] = [0, 255, 0]
    return result_image  # (H, W, 3)

def show_final_output(original, extracted):
    """
    Display original and extracted number image side by side.

    Parameters:
        original (ndarray): Original image of shape (H, W, 3).
        extracted (ndarray): Output image with green number, shape (H, W, 3).
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
    image = Image.open(image_path).convert("RGB")  # PIL image
    image_np = np.array(image)  # (H, W, 3)
    pixels = image_np.reshape((-1, 3)).astype(np.float32)  # (num_samples=H*W, 3)

    # Apply K-means
    labels, centroids = kmeans(pixels, k)  # labels: (H*W,), centroids: (k, 3)
    label_map = labels.reshape(image_np.shape[:2])  # (H, W)

    # Visualize clusters
    visualize_clusters(image_np, label_map, k)

    # Select clusters for the number
    selection = input("Enter cluster indices (comma-separated) that represent the number: ")
    number_clusters = [int(x.strip()) for x in selection.split(",")]

    # Highlight number
    result_image = highlight_number(image_np, label_map, number_clusters)  # (H, W, 3)
    Image.fromarray(result_image).save(f"output_images/{image_id}_number_only.png")

    # Show result
    show_final_output(image_np, result_image)
