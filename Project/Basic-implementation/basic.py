import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from heapq import heappush, heappop
from collections import defaultdict

# Convert image into grayscale using cv2
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image into 8x8 blocks
def convert_to_blocks(image):
    # Resize image to convert it into 8x8 blocks
    image = cv2.resize(image, (np.int32(np.ceil(image.shape[1]/8)*8), np.int32(np.ceil(image.shape[0]/8)*8)))
    blocks = []
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            blocks.append(image[i:i+8, j:j+8])
    return blocks

# Apply DCT to each block
def apply_dct(blocks):
    dct_blocks = []
    for block in blocks:
        dct_blocks.append(dct(dct(block.T, norm='ortho').T, norm='ortho'))
    return dct_blocks

# Quantize the DCT coefficients
def quantize(blocks, quality):
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    quantization_matrix = quantization_matrix * (100 - quality) / 50
    quantized_blocks = []
    for block in blocks:
        quantized_block = np.round(block / quantization_matrix).astype(int)
        quantized_blocks.append(quantized_block)
    return quantized_blocks

# Perform Zigzag scanning to linearize the 8x8 block
def zigzag_scanning(block):
    zigzag = []
    for i in range(8):
        if i % 2 == 0:
            for j in range(i+1):
                zigzag.append(block[j, i-j])
        else:
            for j in range(i+1):
                zigzag.append(block[i-j, j])
    for i in range(1, 8):
        if i % 2 == 0:
            for j in range(8-i):
                zigzag.append(block[7-j, i+j])
        else:
            for j in range(8-i):
                zigzag.append(block[i+j, 7-j])
    return zigzag

# Huffman encoding to compress the linearized data
def huffman_encoding(data):
    from collections import Counter, defaultdict
    from heapq import heappush, heappop, heapify

    frequency = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapify(heap)
    
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    codes = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    huffman_dict = {symbol: code for symbol, code in codes}
    
    encoded_data = "".join([huffman_dict[d] for d in data])
    return encoded_data, huffman_dict

# Apply this compression on ex.png
def compress_image(image, quality):
    image = convert_to_grayscale(image)
    blocks = convert_to_blocks(image)
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize(dct_blocks, quality)
    zigzag_blocks = []
    for block in quantized_blocks:
        zigzag_blocks.append(zigzag_scanning(block))
    encoded_data = ""
    huffman_dict = None
    for block in zigzag_blocks:
        encoded, huffman_dict = huffman_encoding(block)
        encoded_data += encoded
    return encoded_data, huffman_dict

# Huffman decoding to decompress the encoded data
def huffman_decoding(encoded_data, huffman_dict):
    reverse_huffman_dict = {v: k for k, v in huffman_dict.items()}
    decoded_data = []
    code = ""
    for bit in encoded_data:
        code += bit
        if code in reverse_huffman_dict:
            decoded_data.append(reverse_huffman_dict[code])
            code = ""
    return decoded_data

# Perform inverse zigzag scanning to convert a 1D array back to a 2D block
def inverse_zigzag_scanning(input, vmax, hmax):
    output = np.zeros((vmax, hmax), dtype=int)
    i = j = 0
    for k in range(len(input)):
        output[i, j] = input[k]
        if (i + j) % 2 == 0:  # Even stripes
            if j < hmax - 1:
                j += 1
            elif i < vmax - 1:
                i += 1
            else:
                break
            if i > 0:
                i -= 1
        else:  # Odd stripes
            if i < vmax - 1:
                i += 1
            elif j < hmax - 1:
                j += 1
            else:
                break
            if j > 0:
                j -= 1
    return output

# Dequantize the quantized DCT coefficients
def dequantize(blocks, quality):
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    quantization_matrix = quantization_matrix * (100 - quality) / 50
    dequantized_blocks = []
    for block in blocks:
        dequantized_block = block * quantization_matrix
        dequantized_blocks.append(dequantized_block)
    return dequantized_blocks

# Apply Inverse Discrete Cosine Transform (IDCT) to each block
def apply_idct(blocks):
    idct_blocks = []
    for block in blocks:
        idct_blocks.append(idct(idct(block.T, norm='ortho').T, norm='ortho'))
    return idct_blocks

# Reconstruct the image from the encoded data
def reconstruct_image(encoded_data, huffman_dict, image_shape, quality):
    block_size = 8
    zigzag_blocks = []
    i = 0
    while i < len(encoded_data):
        j = i + block_size * block_size
        zigzag_blocks.append(huffman_decoding(encoded_data[i:j], huffman_dict))
        i = j
    blocks = []
    for zigzag_block in zigzag_blocks:
        block = inverse_zigzag_scanning(zigzag_block, block_size, block_size)
        blocks.append(block)
    dequantized_blocks = dequantize(blocks, quality)
    idct_blocks = apply_idct(dequantized_blocks)
    reconstructed_image = np.zeros(image_shape)
    k = 0
    for i in range(0, image_shape[0], block_size):
        for j in range(0, image_shape[1], block_size):
            if k < len(idct_blocks):
                reconstructed_image[i:i+block_size, j:j+block_size] = idct_blocks[k]
                k += 1
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    return np.uint8(reconstructed_image)

# Example usage
image = cv2.imread("ex.png")
encoded_data, huffman_dict = compress_image(image, 50)
image_shape = image.shape[:2]
reconstructed_image = reconstruct_image(encoded_data, huffman_dict, image_shape, 50)
cv2.imwrite("reconstructed_image.png", reconstructed_image)

# Display the original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()