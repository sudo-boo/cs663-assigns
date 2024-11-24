import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
import glob


def compress_block_pca(block, num_components):
    """
    Compress a block using PCA.

    Parameters:
    - block: 2D NumPy array representing a grayscale block.
    - num_components: Number of principal components to retain.

    Returns:
    - compressed_block: Reconstructed block after PCA compression.
    """
    # Step 1: Mean normalization
    mean_block = np.mean(block, axis=1, keepdims=True)
    centered_block = block - mean_block

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(centered_block)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Retain the top `num_components` eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Step 6: Project the block into the PCA subspace
    transformed_data = np.dot(principal_components.T, centered_block)

    # Step 7: Reconstruct the block
    reconstructed_block = np.dot(principal_components, transformed_data) + mean_block

    return reconstructed_block


def apply_local_pca(image, block_size, num_components):
    """
    Apply PCA locally to an image in blocks of size `block_size`.

    Parameters:
    - image: 2D NumPy array representing the grayscale image.
    - block_size: Size of the square block (e.g., 8 for 8x8 blocks).
    - num_components: Number of principal components to retain.

    Returns:
    - compressed_image: Image reconstructed after applying PCA locally.
    """
    compressed_image = np.zeros_like(image)
    h, w = image.shape

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Extract the current block
            block = image[i:i + block_size, j:j + block_size]

            # Handle edge cases where the block size is smaller (near boundaries)
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue

            # Apply PCA to the block
            compressed_block = compress_block_pca(block, num_components)

            # Store the compressed block back in the image
            compressed_image[i:i + block_size, j:j + block_size] = compressed_block

    return compressed_image


def bpp(image_path,img_name, num_components, class_name):
    """
    Calculate the bits per pixel (BPP) for the compressed image.

    Parameters:
    - img_name: Name of the image file.
    - num_components: Number of PCA components used for compression.
    - class_name: Folder name for the image class.

    Returns:
    - bpp: Bits per pixel for the compressed image.
    """
    compressed_image_path = f"output/compressed/{class_name}/{img_name}/{num_components}.png"
    compressed_image = io.imread(compressed_image_path)

    # Calculate the number of bits per pixel
    file_size_bits = os.path.getsize(compressed_image_path) * 8
    image_area = compressed_image.shape[0] * compressed_image.shape[1]
    bpp_value = file_size_bits / image_area

    return bpp_value


def rmse(imageA, imageB):
    """
    Calculate the Root Mean Squared Error (RMSE) between two images.

    Parameters:
    - imageA: First image (original).
    - imageB: Second image (compressed).

    Returns:
    - RMSE: Root mean squared error.
    """
    return np.sqrt(np.mean((imageA - imageB) ** 2))


def main(image_path, img_name, num_components, block_size, class_name):
    image = io.imread(image_path)
    if image.ndim == 3:  # Convert to grayscale if it's an RGB image
        image = color.rgb2gray(image)
    image = image.astype(float)

    # Compress the image using local PCA
    compressed_image = apply_local_pca(image, block_size, num_components)
    image = image.astype(np.uint8)
    compressed_image = compressed_image.astype(np.uint8)

    # Create output directories
    os.makedirs(f"output/original/{class_name}", exist_ok=True)
    os.makedirs(f"output/compressed/{class_name}/{img_name}", exist_ok=True)

    # Save the images
    io.imsave(f"output/original/{class_name}/{img_name}.png", image)
    io.imsave(f"output/compressed/{class_name}/{img_name}/{num_components}.png", compressed_image)


if __name__ == "__main__":
    imgs_path = "../Microsoft-Database/pca-gray/"
    block_size = 8  # Block size for local PCA
    num_components_list = [30, 40, 50, 60, 70, 80, 90]

    buildings = glob.glob(imgs_path + "buildings/*.png")

    plt.figure(figsize=(25, 25))
    for i, img_path in enumerate(buildings):
        print(f"Processing image {i+1}...")
        img_name = f"img{i+1}"
        bpp_buildings = []
        rmse_buildings = []
        for num_components in num_components_list:
            print(f"\twith {num_components} components...")
            main(img_path, img_name, num_components, block_size, "buildings")
            bpp_buildings.append(bpp(img_path, img_name, num_components, "buildings"))
            rmse_buildings.append(rmse(
                io.imread(f"output/original/buildings/{img_name}.png"),
                io.imread(f"output/compressed/buildings/{img_name}/{num_components}.png")
            ))
        plt.plot(bpp_buildings, rmse_buildings, label=f"img{i+1}", marker='o', markersize=10)
    
    print("Processing complete!")

    plt.xlabel("BPP", fontsize=20)
    plt.ylabel("RMSE", fontsize=20)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("output/plot.png")
