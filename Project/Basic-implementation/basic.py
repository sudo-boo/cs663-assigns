import numpy as np
import cv2
from scipy.fftpack import dct, idct
from heapq import heappush, heappop
from collections import defaultdict
import matplotlib.pyplot as plt
import os

def block_dct(block):
    """Apply 2D DCT on an image block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block):
    """Apply 2D inverse DCT on an image block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def quantize(block, quality_factor):
    """Quantize the DCT coefficients."""
    scale = 50 / quality_factor if quality_factor < 50 else 2 - quality_factor / 50
    scaled_Q = np.clip(Q * scale, 1, 255)
    return np.round(block / scaled_Q).astype(int), scaled_Q

def dequantize(block, scaled_Q):
    """Dequantize the DCT coefficients."""
    return block * scaled_Q

def build_huffman_tree(frequencies):
    """Build a Huffman tree given the symbol frequencies."""
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    while len(heap) > 1:
        heappush(heap, [heappop(heap)[0] + heappop(heap)[0]] + [heappop(heap), heappop(heap)])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encoding(data):
    """Perform Huffman encoding on the input data."""
    frequency = defaultdict(int)
    for symbol in data:
        frequency[symbol] += 1
    huffman_tree = build_huffman_tree(frequency)
    huffman_dict = {item[0]: item[1] for item in huffman_tree}
    encoded_data = "".join(huffman_dict[symbol] for symbol in data)
    return encoded_data, huffman_dict

def huffman_decoding(encoded_data, huffman_dict):
    """Decode Huffman encoded data."""
    reversed_dict = {v: k for k, v in huffman_dict.items()}
    decoded = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reversed_dict:
            decoded.append(reversed_dict[buffer])
            buffer = ""
    return decoded

def compress_image(image, block_size, quality_factor):
    h, w = image.shape
    compressed_blocks = []
    scales = []
    
    # Divide image into non-overlapping blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = block_dct(block)
            quantized_block, scaled_Q = quantize(dct_block, quality_factor)
            compressed_blocks.append(quantized_block.flatten())
            scales.append(scaled_Q.flatten())
    
    # Flatten and Huffman encode all blocks
    flat_data = np.hstack(compressed_blocks).flatten()
    encoded_data, huffman_dict = huffman_encoding(flat_data)
    return encoded_data, huffman_dict, h, w, scales

def decompress_image(encoded_data, huffman_dict, h, w, scales, block_size):
    decoded_data = huffman_decoding(encoded_data, huffman_dict)
    decompressed_image = np.zeros((h, w))
    
    # Decode blocks
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            quantized_block = np.array(decoded_data[idx:idx + block_size**2]).reshape(block_size, block_size)
            scaled_Q = np.array(scales[idx // block_size**2]).reshape(block_size, block_size)
            idct_block = block_idct(dequantize(quantized_block, scaled_Q))
            decompressed_image[i:i+block_size, j:j+block_size] = idct_block
            idx += block_size**2
    
    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

def calculate_rmse(original, compressed):
    return np.sqrt(np.mean((original - compressed) ** 2))

def calculate_bpp(compressed_size, image_shape):
    total_pixels = image_shape[0] * image_shape[1]
    return compressed_size / total_pixels

def main():
    # Parameters
    block_size = 8
    quality_factors = [10, 20, 30, 50, 70, 90]  # Test multiple quality factors
    
    # Load grayscale image
    image = cv2.imread("C:\Users\surya\OneDrive - Indian Institute of Technology Bombay\docmain\sem5\cs663\cs663-assigns\Project\Microsoft-Database\sheep\108_0890.JPG", cv2.IMREAD_GRAYSCALE)
    
    rmse_results = []
    bpp_results = []
    
    for quality in quality_factors:
        encoded_data, huffman_dict, h, w, scales = compress_image(image, block_size, quality)
        compressed_size = len(encoded_data)
        
        decompressed_image = decompress_image(encoded_data, huffman_dict, h, w, scales, block_size)
        
        rmse = calculate_rmse(image, decompressed_image)
        bpp = calculate_bpp(compressed_size, image.shape)
        
        rmse_results.append(rmse)
        bpp_results.append(bpp)
        
        # Save decompressed image for visualization
        cv2.imwrite(f"decompressed_q{quality}.png", decompressed_image)
    
    # Plot RMSE vs BPP
    plt.figure()
    plt.plot(bpp_results, rmse_results, marker='o')
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs BPP")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
