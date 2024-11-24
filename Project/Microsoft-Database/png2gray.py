import cv2
import os

# converting all png images to grayscale
def png2gray():
    for file in os.listdir("colored-imgs"):
        if file.endswith(".png"):
            img = cv2.imread(f"colored-imgs/{file}", cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            os.makedirs("gray-imgs", exist_ok=True)
            cv2.imwrite(f"gray-imgs/{file}", gray)
            print(f"Converted {file} to grayscale.")

if __name__ == "__main__":
    png2gray()