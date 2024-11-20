import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from PIL import Image

def load_image(path):
    """
    Load an image and convert it to grayscale.
    
    Args:
        path (str): Path to the image file.
    
    Returns:
        np.ndarray: Grayscale image as a 2D numpy array.
    """
    image = Image.open(path).convert("L")  # Convert to grayscale
    return np.array(image)

def split_into_blocks(image, block_size):
    """
    Split an image into non-overlapping blocks.
    
    Args:
        image (np.ndarray): Input 2D image array.
        block_size (int): Size of the square blocks.
    
    Returns:
        np.ndarray: Array of blocks reshaped to (num_blocks, block_size, block_size).
    """
    h, w = image.shape
    assert h % block_size == 0 and w % block_size == 0, "Image dimensions must be divisible by block size"
    blocks = [
        image[i:i+block_size, j:j+block_size]
        for i in range(0, h, block_size)
        for j in range(0, w, block_size)
    ]
    return np.array(blocks)

def psnr(original, reconstructed):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (np.ndarray): Original image array.
        reconstructed (np.ndarray): Reconstructed image array.
    
    Returns:
        float: PSNR value in dB.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:  # Avoid log(0)
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def evaluate_compression(original, reconstructed):
    """
    Evaluate the compression performance by comparing the original and reconstructed images.
    
    Args:
        original (np.ndarray): Original image array.
        reconstructed (np.ndarray): Reconstructed image array.
    
    Returns:
        None
    """
    psnr_value = psnr(original, reconstructed)
    print(f"PSNR: {psnr_value:.2f} dB")


# Helper function: Thresholding
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Step 1: Fix G, optimize coefficients
def optimize_coefficients(G, x, lambda_):
    projections = G.T @ x
    return soft_threshold(projections, np.sqrt(lambda_))

# Step 2: Fix coefficients, optimize G
def optimize_transform(x, coefficients):
    Y = np.mean(np.outer(coefficients[:, i], x[:, i]) for i in range(x.shape[1]))
    U, _, Vt = svd(Y)
    return Vt.T @ U.T

def annealing_optimization(x, G_init, lambdas):
    G = G_init
    for lambda_ in lambdas:
        coefficients = np.array([optimize_coefficients(G, xi, lambda_) for xi in x.T])
        G = optimize_transform(x, coefficients)
    return G

def classify_blocks(x, num_clusters):
    # Compute features (e.g., gradient direction)
    gradients = np.gradient(x, axis=(0, 1))
    features = np.sqrt(gradients[0]**2 + gradients[1]**2).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters).fit(features)
    return kmeans.labels_.reshape(x.shape)

def compress_image(image, transforms, block_size):
    h, w = image.shape
    compressed_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            transform = select_transform(block, transforms)
            coefficients = transform.T @ block.flatten()
            compressed_blocks.append((coefficients, transform))
    return compressed_blocks

def decompress_image(compressed_blocks, block_size, h, w):
    decompressed = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            coefficients, transform = compressed_blocks[idx]
            block = transform @ coefficients
            decompressed[i:i+block_size, j:j+block_size] = block.reshape(block_size, block_size)
            idx += 1
    return decompressed

# Load and preprocess the image
image = load_image('ex.png')  # Replace with actual image loader

# Parameters
block_size = 8
num_clusters = 8
lambdas = [1000, 500, 100, 10]  # Annealing parameters

# Step 1: Preprocess into blocks and cluster
blocks = split_into_blocks(image, block_size)
labels = classify_blocks(blocks, num_clusters)

# Step 2: Design SOTs for each cluster
G_init = np.eye(block_size * block_size)
transforms = [annealing_optimization(blocks[labels == i], G_init, lambdas) for i in range(num_clusters)]

# Step 3: Compress the image
compressed = compress_image(image, transforms, block_size)

# Step 4: Decompress and evaluate
reconstructed = decompress_image(compressed, block_size, *image.shape)
evaluate_compression(image, reconstructed)
