from PIL import Image
import os
import glob

def convert_to_grayscale(input_path, output_path):
    """
    Convert a colored PNG image to grayscale.

    Parameters:
    - input_path: Path to the input colored PNG image.
    - output_path: Path to save the grayscale image.
    """
    try:
        # Open the colored image
        img = Image.open(input_path)
        
        # Convert to grayscale
        grayscale_img = img.convert("L")
        
        # Save the grayscale image
        grayscale_img.save(output_path)
        print(f"Grayscale image saved at: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_path = "pca-colored/flowers/*.png"
    output_path = "pca-gray/flowers/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Convert all colored images to grayscale
    for i, img_path in enumerate(glob.glob(input_path)):
        img_name = f"img{i+1}"
        convert_to_grayscale(img_path, f"{output_path}/{img_name}.png")

    input_path2 = "pca-colored/buildings/*.png"
    output_path2 = "pca-gray/buildings/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_path2, exist_ok=True)

    # Convert all colored images to grayscale
    for i, img_path in enumerate(glob.glob(input_path2)):
        img_name = f"img{i+1}"
        convert_to_grayscale(img_path, f"{output_path2}/{img_name}.png")