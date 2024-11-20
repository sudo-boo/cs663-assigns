import cv2
import numpy as np
from scipy.ndimage import laplace
from scipy.sparse import linalg
from scipy.sparse import diags

def marr_hildreth_edge_detection(image, sigma=5):
    """
    Detect edges using the Marr-Hildreth (Laplacian of Gaussian) method.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # print("blurred", blurred)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    # print("log", log)
    edges = np.where(np.abs(log) < 0.01, 0, 1)  # Zero-crossings as edges
    return edges.astype(np.uint8)

import numpy as np

def hysteresis_thresholding(edges, low_threshold=20, high_threshold=50):
    """
    Apply hysteresis thresholding to refine edge detection.
    """
    strong = edges > high_threshold
    weak = (edges <= high_threshold) & (edges > low_threshold)

    result = np.zeros_like(edges)
    result[strong] = 255

    def recursive_connect(y, x):
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1] and weak[ny, nx]:
                result[ny, nx] = 255
                weak[ny, nx] = False  # Mark as visited
                recursive_connect(ny, nx)

    for y, x in np.argwhere(strong):
        recursive_connect(y, x)

    return result

def quantize_image(image, levels=16):
    """
    Quantize the image into a specified number of levels.
    """
    max_val = image.max()
    min_val = image.min()
    step = (max_val - min_val) / (levels - 1)
    quantized_image = np.round((image - min_val) / step) * step + min_val
    return quantized_image.astype(np.uint8)

def homogeneous_diffusion_inpainting(edge_image, pixel_values, iterations=500):
    """
    Perform homogeneous diffusion to inpaint the missing data.
    Solve the steady-state Laplace equation using finite differences.
    """
    h, w = edge_image.shape
    grid = np.zeros((h, w), dtype=float)
    grid[edge_image > 0] = pixel_values[edge_image > 0]

    for _ in range(iterations):
        laplacian = (
            np.roll(grid, 1, axis=0) +
            np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1) -
            4 * grid
        )
        grid[edge_image == 0] += 0.25 * laplacian[edge_image == 0]

    return grid

def compress_image(image, quantization_levels=16):
    """
    Compress the input image by encoding edges and adjacent pixel values.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = marr_hildreth_edge_detection(gray_image)
    refined_edges = hysteresis_thresholding(edges)

    quantized_image = quantize_image(gray_image, levels=quantization_levels)
    # print("refined_edges", refined_edges)
    print("edges", edges)

    compressed = {
        "edges": refined_edges,
        "quantized": quantized_image[refined_edges > 0]
    }

    return compressed

def decompress_image(compressed, original_shape):
    """
    Decompress the image by reconstructing using homogeneous diffusion.
    """
    edge_image = compressed["edges"]
    pixel_values = compressed["quantized"]
    return homogeneous_diffusion_inpainting(edge_image, pixel_values).reshape(original_shape)

# Example Usage
if __name__ == "__main__":
    image = cv2.imread("ex.png", cv2.IMREAD_GRAYSCALE)

    # Compress the image
    compressed_data = compress_image(image)

    # Decompress the image
    # print(compressed_data["quantized"])
    reconstructed_image = decompress_image(compressed_data, image.shape)

    # Save or display the reconstructed image
    cv2.imwrite("reconstructed_image.png", reconstructed_image)
    cv2.imshow("Original", image)
    cv2.imshow("Reconstructed", reconstructed_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
