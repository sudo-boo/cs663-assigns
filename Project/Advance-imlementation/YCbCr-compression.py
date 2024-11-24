import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
import glob
import os


# Standard JPEG Quantization Table for Luminance
BASE_QTABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

def bgr2ycbcr(img):
    """Converts a BGR image to YCbCr color space.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        np.ndarray: YCbCr image.
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return ycbcr

def ycbcr2bgr(img):
    """Converts a YCbCr image to BGR color space.

    Args:
        img (np.ndarray): YCbCr image.

    Returns:
        np.ndarray: BGR image.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    return bgr


def calculate_rmse(original, compressed):
    """Compute the Root Mean Squared Error (RMSE) between two images."""
    return np.sqrt(np.mean((original - compressed) ** 2))


def calculate_bpp(file_size_bytes, image_shape):
    """Compute Bits Per Pixel (BPP) for the compressed image."""
    total_pixels = image_shape[0] * image_shape[1]
    return (file_size_bytes * 8) / total_pixels


def get_DCT_matrix(size):
    """Returns the normalised square matrix, used in the computation of the Discrete Cosine Transform

    Argument:
    size: the size of the matrix, which is the size of each basis vector. This must be equal to the length
            of the signal to be transformed, obviously

    Returns:
    DCT_matrix: The required DCT matrix
    """
    # initializing x and m
    x = np.arange(start=0, stop=size).reshape(1, size)  # row vector
    x = 2 * x + 1
    m = np.arange(start=0, stop=size).reshape(size, 1)  # column vector

    # the outer product
    matrix = np.dot(m, x)

    # taking the cosine
    DCT_matrix = np.cos(matrix * np.pi / 2 / size)

    # normalizing, dividing each basis by its magnitude
    # DCT_matrix[0, :] *= np.sqrt(1/size)
    # DCT_matrix[1:,:] *= np.sqrt(2/size)

    return DCT_matrix


DCT_MATRIX = get_DCT_matrix(8)


def DCT_2D(in_block):
    """computes the DCT transform of the incoming signal
    Arguments:
    in_block: the input 2D signal


    Returns:
    dct_coeff: The normalised DCT coefficients corresponding to this block
    """

    # First some input validation
    row, col = in_block.shape
    assert row == 8 and col == 8

    # DCT computation
    Intermediate = np.dot(DCT_MATRIX, in_block)
    dct_coeff = np.dot(Intermediate, DCT_MATRIX.T)

    # scaling
    dct_coeff[0, 0] /= 64
    dct_coeff[0, 1:] /= 32
    dct_coeff[1:, 0] /= 32
    dct_coeff[1:, 1:] /= 16

    return dct_coeff


def IDCT_2D(in_block):
    """computes the IDCT transform of the incoming DCT coefficients block
    Arguments:
    in_block: the input 2D DCT coefficients


    Returns:
    inverted: The result of the IDCT
    """
    # First some input validation
    row, col = in_block.shape
    assert row == 8 and col == 8

    # DCT computation

    Intermediate = np.dot(DCT_MATRIX.T, in_block)
    inverted = np.dot(Intermediate, DCT_MATRIX)

    return inverted


def generate_qtable(base_qtable, q_factor):
    """
    Generate a JPEG quantization table for a given quality factor.

    Args:
        base_qtable (np.ndarray): Base quantization table.
        q_factor (int): Quality factor (1 to 100).

    Returns:
        np.ndarray: Scaled quantization table.
    """
    # Compute scaling factor
    scale = 50 / q_factor if q_factor < 50 else 2 - q_factor / 50
    # Scale and clip the quantization table
    scaled_qtable = np.clip(base_qtable * scale, 1, 255)
    return np.round(scaled_qtable).astype(int)


def divide_quant(block, q_table):
    """Divides the each element in the block by the corresponding element in the Quantization table, then rounds

    Arguments:
    block: The block to be divided by the table
    q_table: The Quantization table "ones by default"

    Returns: The result of dividing, then rounding
    """
    res = block / q_table
    res = res.astype(np.int64)  # rounding
    return res


# At the decoder, we multiply by the Q_table
def multiply_quant(block, q_table):
    return block * q_table


def get_ZigZag_indices(size):
    """Returns the order of indeces to be parsed in zigzag parsing

    Arguments:
    size: The size of the square matrix to be parsed

    Returns:
    indeces: indeces ordered according to zigzag parsing
    """

    # this will hold the first half
    indeces = []
    # this will hold the second half indeces
    reversed_indeces = []

    for i in range(size):
        for j in range(i + 1):
            # if odd
            if i % 2 != 0:
                indeces.append((j, i - j))
                reversed_indeces.append((size - 1 - j, size - 1 - i + j))
            # if even
            else:
                indeces.append((i - j, j))
                reversed_indeces.append((size - 1 - i + j, size - 1 - j))

    # reverse
    reversed_indeces = reversed_indeces[::-1]
    # exclude the main diagonal part because it is repeated
    indeces = indeces[:-size]
    # merge the two lists
    indeces.extend(reversed_indeces)

    return indeces


# the ZIGZAG indices will be a global variable.
ZIG_ZAG_INDICES = get_ZigZag_indices(8)
ZIG_ZAG_ROW, ZIG_ZAG_COL = zip(*ZIG_ZAG_INDICES)


def encode_runlength(array):
    """Returns the encoded array using runlength code
    The function will apply runlength to "0" because it is the most occuring element

    Arguments:
    array: a 1D numpy array to be encoded

    Returns: a 1D list, encoded using runlength
    """
    encoded = []
    counter = 0

    # this for loop will handle zeros that are interrupted before array ends
    for element in array:
        if element == 0:  # if element is zero, increment counter
            counter += 1
        else:  # if not, check if we were in a zero sequence
            if counter != 0:
                # zero sequence interrupted
                encoded.extend([0, counter])
                # reset counter
                counter = 0

            # add the non zero element to the array
            encoded.append(element)

    # This handles trailing zeros
    if counter != 0:
        encoded.extend([0, counter])
    return encoded


def decode_runlength(array):
    """Decodes the runlength code of zeros
    Argument:
    array: a 1D numpy array of runlength code

    Returns: The original 1D sequence as a numpy array
    """
    decoded = []
    index = 0
    while index < len(array):
        if array[index] == 0:  # zero detected
            count = array[index + 1]
            # recreate the zero sequence
            zero_seq = [0] * count
            # append the zero sequence
            decoded.extend(zero_seq)
            # update the index by 2 to bypass the count
            index += 2
        else:  # not a zero
            decoded.append(array[index])
            index += 1
    return np.array(decoded)


def pad_image(image):
    """Zero pads the image if necessary to be composed of 8x8 blocks

    Arguments:
    image: a 2D numpy matrix

    Returns:
    padded_image: a padded image with both dimensions as multiples of 8
    """

    # initializing the padded dims to rows and cols
    row_padded, col_padded = image.shape

    # padding the rows if they are not multiples of 8
    if row_padded % 8 != 0:
        row_padded = row_padded + (8 - row_padded % 8)
    # padding the cols
    if col_padded % 8 != 0:
        col_padded = col_padded + (8 - col_padded % 8)

    # if no padding happened, return the original image
    if (row_padded, col_padded) == image.shape:
        return image

    (row, col) = image.shape
    # initialize with new dims
    padded_image = np.zeros((row_padded, col_padded))

    # assign the old image to its right place
    padded_image[0:row, 0:col] = image

    return padded_image


def blockify(image):
    """Returns a list of image blocks
    Arguments:
    image: a 2D matrix representing a matrix

    Returns
    blocks: a list of 8x8 blocks
    """
    # zero pad if necessary
    image_padded = pad_image(image)
    # initialize the blocks list
    blocks = []
    # find out the number of blocks across each dim
    row, col = image_padded.shape
    n_block_row = int(row / 8)
    n_block_col = int(col / 8)
    # unenroll
    for i in range(n_block_row):
        for j in range(n_block_col):
            current_block = image_padded[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            blocks.append(current_block)

    return blocks, image_padded.shape


def encode_JPEG(image, q_table):
    """Performs JPEG compression on the image

    Arguments:
    image: a 2D matrix representing the grayscale image
    q_table: The qunatization table used while compressing

    Returns:
    huff_stream: The stream of bits obtained from the Huffman encoding
    huff_tree: The tree that holds the coding scheme. Essential for decoding
    huff_dict: A dictionary which maps each symbol to its assigned code by the Huffman code
    padded_shape: The size of the padded image. Will be usefull for reconstructing the image
    """
    # dividing into blocks
    blocks, padded_shape = blockify(image)

    # initializing the stream
    stream = []
    # for each block
    for block in blocks:
        # applying 2D DCT
        DCT_block = DCT_2D(block)

        # dividing by quantization matrix
        q_block = divide_quant(DCT_block, q_table)

        # zigzag unrolling or spreading
        vector = q_block[ZIG_ZAG_ROW, ZIG_ZAG_COL]

        # Runlength
        rn_vector = encode_runlength(vector)

        # appending to stream
        stream.extend(rn_vector)

    # Now that we have the stream, apply huffman encoding
    huff_stream, huff_tree, huff_dict, symb_dict = encode_huffman(stream)

    return huff_stream, huff_tree, huff_dict, symb_dict, padded_shape


def decode_jpeg(hf_stream, hf_tree, image_dims, q_table):
    """Decodes a Huffman stream into a 2D representation of grayscale image
    Arguments:
    hf_stream: The Stream of bits obtained from huffman code
    hf_tree: The Huffman tree to be used in decoding
    image_dims: The dimension of the image to be decoded
    q_table: The qunatization table that will be multipllied by each block before IDCT

    Returns:
    ret_image: The 2D decoded image
    """
    # init the returned image
    ret_image = np.empty(image_dims)

    # The number of blocks along the columns
    n_block_col = int(image_dims[1] / 8)

    # decode the huffman stream
    dec_hf = decode_huffman(hf_stream, hf_tree)

    # runlength decode
    dec_rn = decode_runlength(dec_hf)

    # Divide the decoded runlength into chunks of 64 elements
    # each rolled chunk will be multiplied by q_table and grouped into a decoded image
    n_chunks = int(len(dec_rn) / 64)

    for i in range(n_chunks):
        chunk = dec_rn[i * 64 : (i + 1) * 64]

        # roll it into a 2D matrix
        rolled = np.empty((8, 8))
        rolled[ZIG_ZAG_ROW, ZIG_ZAG_COL] = chunk

        # multiply by q_table
        rolled = rolled * q_table

        # IDCT and grouping
        row_index = int(i / n_block_col)
        col_index = i % n_block_col
        ret_image[
            row_index * 8 : (row_index + 1) * 8, col_index * 8 : (col_index + 1) * 8
        ] = IDCT_2D(rolled)
        # resize the image to the original size
        ret_image = ret_image[: image_dims[0], : image_dims[1]]

    return ret_image


def rmse_bpp(original, q_tables, image_dims, output_dir):
    """Compute the RMSE and BPP for different quality factors.
    Arguments:
    original: The original image.
    q_tables: A dictionary of quality factors and their corresponding quantization tables.
    image_dims: The dimensions of the original image.
    output_dir: The directory to save the compressed images.

    Returns:
    bpps: List of bits per pixel for each quality factor.
    rmses: List of RMSE values for each quality factor.

    """

    bpps = []
    rmses = []
    for q_factor, q_table in q_tables.items():
        hf_stream, hf_tree, hf_dict, symb_dict, dims = encode_JPEG(original, q_table)
        decoded = decode_jpeg(hf_stream, hf_tree, dims, q_table)
        path = os.path.join(output_dir, f"{q_factor}.jpg")
        cv2.imwrite(path, decoded)
        size = os.path.getsize(path)
        rmse = calculate_rmse(original, decoded)
        bpp = calculate_bpp(size, image_dims)

        bpps.append(bpp)
        rmses.append(rmse)

    return bpps, rmses




# downsampling Cb and Cr channels
def downsample(img):
    """
    Downsample the Cb and Cr channels by a factor of 2.
    """
    y, cb, cr = cv2.split(img)
    cb = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2))
    cr = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2))
    cb = cv2.resize(cb, (y.shape[1], y.shape[0]))
    cr = cv2.resize(cr, (y.shape[1], y.shape[0]))
    
    return cv2.merge((y, cb, cr))

# upsampling Cb and Cr channels
# def upsample(img):
#     """Upsamples the Cb and Cr channels of a YCbCr image by a factor of 2.

#     Args:
#         img (np.ndarray): YCbCr image.

#     Returns:
#         np.ndarray: Upsampled YCbCr image.
#     """
#     y, cb, cr = cv2.split(img)
#     cb = cv2.resize(cb, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     cr = cv2.resize(cr, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     return cv2.merge((y, cb, cr))



def main():
    # load the images from Microsodt-Database folder
    dir_path = "../Microsoft-Database/"
    files = glob.glob(dir_path + "flowers/*.JPG")
    img_path = "output/compressed_imgs"
    original_path = "output/original_imgs"
    plot_path = "output/plot.png"

    # read the images in rgb
    imgs = [cv2.imread(file) for file in files]

    q_factors = [
        5,
        10,
    ]
    q_tables = {q: generate_qtable(BASE_QTABLE, q) for q in q_factors}

    os.makedirs(img_path, exist_ok=True)
    os.makedirs(original_path, exist_ok=True)

    plt.figure(figsize=(25, 25))
    for i in range(len(imgs)):
        img = imgs[i]
        img_name = f"img{i+1}"
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(f"{original_path}/{img_name}.jpg", img)
        img = bgr2ycbcr(img)
        img = downsample(img)
    
        y = img[:, :, 0]
        cb = img[:, :, 1]
        cr = img[:, :, 2]
        
        # print(y.shape, cb.shape, cr.shape)
        print(img)
        print('\n')

        os.makedirs(f"{img_path}/{img_name}", exist_ok=True)
        os.makedirs(f"{img_path}/{img_name}/y", exist_ok=True)
        os.makedirs(f"{img_path}/{img_name}/cb", exist_ok=True)
        os.makedirs(f"{img_path}/{img_name}/cr", exist_ok=True)

        bpps_y, rmses_y = rmse_bpp(y, q_tables, y.shape, f"{img_path}/{img_name}/y")
        bpps_cb, rmses_cb = rmse_bpp(cb, q_tables, cb.shape, f"{img_path}/{img_name}/cb")
        bpps_cr, rmses_cr = rmse_bpp(cr, q_tables, cr.shape, f"{img_path}/{img_name}/cr")

    # read y,cb and cr images
    y_imgs = glob.glob(f"{img_path}/{img_name}/y/*.jpg")
    cb_imgs = glob.glob(f"{img_path}/{img_name}/cb/*.jpg")
    cr_imgs = glob.glob(f"{img_path}/{img_name}/cr/*.jpg")

    y_files = []
    cb_files = []
    cr_files = []
    for i in range(len(y_imgs)):
        y_files.append(cv2.imread(y_imgs[i], cv2.IMREAD_GRAYSCALE))
        cb_files.append(cv2.imread(cb_imgs[i], cv2.IMREAD_GRAYSCALE))
        cr_files.append(cv2.imread(cr_imgs[i], cv2.IMREAD_GRAYSCALE))
    
    for y,cb,cr in zip(y_files,cb_files,cr_files):
        i = 0
        while i < 21:
            img_name = f"img{i+1}"
            y = cv2.resize(y, (256, 256))
            cb = cv2.resize(cb, (256, 256))
            cr = cv2.resize(cr, (256, 256))

            img = cv2.merge((y, cb, cr))
            img = ycbcr2bgr(img)
            mpimg.imsave(f"{img_path}/{img_name}.jpg", img)
            i += 1 
        

    # plottning rmse vs bpp for every file corresponding to every quality factor
    for i in range(len(imgs)):
        compressed_path = glob.glob(f"{img_path}/img{i+1}/*.jpg")
        # original_path = glob.glob(f"{original_path}/img{i+1}.jpg")

        compressed_imgs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in compressed_path]
        original_img = cv2.imread(f"{original_path}/img{i+1}.jpg", cv2.IMREAD_GRAYSCALE)

        rmses = []
        bpps = []
        for j in range(len(compressed_imgs)):
            compressed = compressed_imgs[j]
            original = original_img
            rmse = calculate_rmse(original, compressed)
            bpp = calculate_bpp(os.path.getsize(compressed_path[j]), original.shape)
            rmses.append(rmse)
            bpps.append(bpp)
        
        plt.plot(bpps, rmses, label=f"img{i+1}", marker="o", markersize=15)

    plt.xlabel("Bits Per Pixel", fontsize=30)
    plt.ylabel("RMSE", fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig(plot_path)



if __name__ == "__main__":
    main()
