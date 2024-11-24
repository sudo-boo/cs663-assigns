import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
import glob

def compress_image_pca(image, num_components):
    """
    Compress a grayscale image using PCA.

    Parameters:
    - image: 2D NumPy array representing the grayscale image.
    - num_components: Number of principal components to retain.

    Returns:
    - compressed_image: Reconstructed image after PCA compression.
    - variance_explained: Percentage of variance explained by the retained components.
    """
    # Step 1: Mean normalization
    mean_image = np.mean(image, axis=1, keepdims=True)
    centered_image = image - mean_image

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(centered_image)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Retain the top `num_components` eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Step 6: Project the image data into the PCA subspace
    transformed_data = np.dot(principal_components.T, centered_image)

    # Step 7: Reconstruct the image
    compressed_image = np.dot(principal_components, transformed_data) + mean_image

    # Calculate variance explained
    variance_explained = np.sum(eigenvalues[:num_components]) / np.sum(eigenvalues)

    return compressed_image, variance_explained


#calculating rmse between two images
def rmse(imageA, imageB):
    return np.sqrt(np.mean((imageA - imageB) ** 2))

# calculating bpp for the compressed image
def bpp(image_path, img_name, num_components, class_name):
    compressed_image = io.imread(f"output/compressed/{class_name}/{img_name}/{num_components}.png")
    compressed_image = compressed_image.astype(float)

    # Calculate the number of bits per pixel
    bpp = (os.path.getsize(f"output/compressed/{class_name}/{img_name}/{num_components}.png") * 8) / (compressed_image.shape[0] * compressed_image.shape[1])

    return bpp

# Load and preprocess the image
def main(image_path,img_name, num_components,class_name):
    image = io.imread(image_path)
    if image.ndim == 3:  # Convert to grayscale if it's an RGB image
        image = color.rgb2gray(image)
    image = image.astype(float)

    # Compress the image using PCA
    # num_components = 50  # Number of principal components to retain
    compressed_image, variance_explained = compress_image_pca(image, num_components)
    image = image.astype(np.uint8)
    compressed_image = compressed_image.astype(np.uint8)

    # saving the compressed image and orginal image
    os.makedirs("output/compressed", exist_ok=True)
    os.makedirs("output/original", exist_ok=True)
    os.makedirs(f"output/original/{class_name}", exist_ok=True)
    os.makedirs(f"output/compressed/{class_name}/{img_name}", exist_ok=True)


    io.imsave(f"output/compressed/{class_name}/{img_name}/{num_components}.png", compressed_image)
    io.imsave(f"output/original/{class_name}/{img_name}.png", image)


if __name__ == "__main__":
    imgs_path = "../Microsoft-Database/pca-gray/"
    num_components = [30, 40, 50, 60, 70, 80, 90]
    
    buildings = glob.glob(imgs_path + "buildings/*.png")    

    plt.figure(figsize=(25,25))
    for i,img_path in enumerate(buildings):
        img_name = f"img{i+1}"
        bpp_buildings = []
        rmse_buildings = []
        for num in num_components:
            main(img_path, img_name, num, "buildings")
            bpp_buildings.append(bpp(img_path, img_name, num, "buildings"))
            rmse_buildings.append(rmse(io.imread(f"output/original/buildings/{img_name}.png"), io.imread(f"output/compressed/buildings/{img_name}/{num}.png")))
        plt.plot(bpp_buildings, rmse_buildings, label=f"img{i+1}", marker='o', markersize=30)
    plt.xlabel("BPP", fontsize=40)
    plt.ylabel("RMSE", fontsize=40)
    plt.legend(fontsize=35)
    plt.grid()
    plt.savefig("output/plot_buildings.png")
        