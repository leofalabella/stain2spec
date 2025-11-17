import cv2 as cv
import numpy as np
import os 
from sklearn.decomposition import PCA

input_dir = "data/processed/HE"
output_dir = "data/processed/AF"


def pca_grayscale(img):
    img_flat = img.reshape(-1, 3)
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(img_flat).reshape(img.shape[0], img.shape[1])
    return cv.normalize(pca_result, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

for fname in os.listdir(input_dir):
    if not fname.endswith('.png'):
        continue
    path = os.path.join(input_dir, fname)
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB

    # choose one of the transformations:
    gray = pca_grayscale(img)
    # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # Convert to grayscale
    # gray = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 2]  # Convert to HSV
    # gray = cv.cvtColor(img, cv.COLOR_RGB2LAB)[:, :, 0]  # Convert to LAB

    # simulating scattering
    gray = 255 - gray # Invert colors to simulate scattering
    gray = cv.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur

    out = cv.resize(gray, (256, 256))  # Resize to 256x256
    cv.imwrite(os.path.join(output_dir, fname), out)