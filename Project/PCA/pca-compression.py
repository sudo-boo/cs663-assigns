import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.decomposition import PCA
import os
import glob

# applying PCA on the block of the image
def compress_block_pca(block, num_components):
    """
    Compress a block using PCA.

    Parameters:
    - block: 2D NumPy array representing the block.
    - num_components: Number of principal components to retain.

    Returns:
    - block_compressed: Reconstructed block after PCA compression.
    - variance_explained: Percentage of variance explained by the retained components.
    """

    mean_block = np.mean(block, axis=0, keepdims=True)
    centered_block = block - mean_block

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(centered_block)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # print(eigenvalues)
    # print("-----------------------------")

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Retain the top `num_components` eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Step 6: Project the block data into the PCA subspace
    transformed_data = np.dot(principal_components.T, centered_block)

    # Step 7: Reconstruct the block
    block_compressed = np.dot(principal_components, transformed_data) + mean_block

    return block_compressed

# applying PCA on the every block of the image
def compress_image_pca(image, num_components, block_size=16):
    """
    Compress a grayscale image using PCA.

    Parameters:
    - image: 2D NumPy array representing the grayscale image.
    - num_components: Number of principal components to retain.
    - block_size: Size of the blocks to divide the image into.

    Returns:
    - compressed_image: Reconstructed image after PCA compression.
    - variance_explained: Percentage of variance explained by the retained components.
    """
    # Step 1: Divide the image into blocks
    height, width = image.shape
    compressed_image = np.zeros((height, width))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]

            # Ensure the block is of the correct size
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue

            # Step 2: Apply PCA on the block
            block_compressed = compress_block_pca(block, num_components)

            # Step 3: Reconstruct the block
            compressed_image[i:i+block_size, j:j+block_size] = block_compressed



    return compressed_image

#calculating rmse between two images
def rmse(imageA, imageB):
    sum_sq = np.sum((imageA.astype("float")) ** 2)
    forbenius_norm = np.sqrt(sum_sq / float(imageA.shape[0] * imageA.shape[1]))
    return np.sqrt(np.mean((imageA - imageB) ** 2))/forbenius_norm


# calculating bpp for the compressed image
# def bpp(image_path, img_name, num_components, class_name):
#     compressed_image = io.imread(f"output/compressed/{class_name}/{img_name}/{num_components}.png")
#     compressed_image = compressed_image.astype(float)

#     # Calculate the number of bits per pixel
#     bpp = (os.path.getsize(f"output/compressed/{class_name}/{img_name}/{num_components}.png") * 8) / (compressed_image.shape[0] * compressed_image.shape[1])

#     return bpp

# Load and preprocess the image
def main(image_path, img_name, num_components, class_name):
    image = io.imread(image_path)
    if image.ndim == 3:  # Convert to grayscale if it's an RGB image
        image = color.rgb2gray(image)
    image = image.astype(float)

    # Compress the image using PCA
    compressed_image = compress_image_pca(image, num_components)
    image = image.astype(np.uint8)
    compressed_image = compressed_image.astype(np.uint8)

    total_pixels = image.shape[0] * image.shape[1]
    num_blocks = np.ceil(image.shape[0] / 16) * np.ceil(image.shape[1] / 16)
    bpp = ((2*num_components + 1) * num_blocks / total_pixels)*16*8

    # saving the compressed image and original image
    os.makedirs(f"output/compressed/{class_name}/{img_name}", exist_ok=True)
    os.makedirs(f"output/original/{class_name}", exist_ok=True)

    io.imsave(f"output/compressed/{class_name}/{img_name}/{num_components}.png", compressed_image)
    io.imsave(f"output/original/{class_name}/{img_name}.png", image)

    return bpp

if __name__ == "__main__":
    imgs_path = "../Microsoft-Database/pca-gray/"
    num_components = [1,2,4,8]
    
    buildings = glob.glob(imgs_path + "buildings/*.png")    

    plt.figure(figsize=(25,25))
    for i, img_path in enumerate(buildings):
        img_name = f"img{i+1}"
        bpp_buildings = []
        rmse_buildings = []
        for num in num_components:
            bpp = main(img_path, img_name, num, "buildings")
            bpp_buildings.append(bpp)
            rmse_buildings.append(rmse(io.imread(f"output/original/buildings/{img_name}.png"), io.imread(f"output/compressed/buildings/{img_name}/{num}.png")))

        plt.plot(bpp_buildings, rmse_buildings, label=f"img{i+1}", marker='o', markersize=30)
    plt.xlabel("BPP", fontsize=40)
    plt.ylabel("RMSE", fontsize=40)
    plt.legend(fontsize=35)
    plt.grid()
    plt.savefig("output/plot.png")