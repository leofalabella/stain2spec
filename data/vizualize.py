from openslide import OpenSlide
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import pca

slide = OpenSlide("data/raw/test_001.tif")
x, y = 20000, 20000  # Coordinates for the patch
patch = slide.read_region((x, y), level=0, size=(500, 500)).convert('RGB')
patch.save("data/processed/HE_patch_001.png")

# im = Image.open("data/raw/test_001.tif")
# im = im.show()

img = cv.imread("data/processed/HE_patch_001.png")
img = cv.resize(img, (256, 256))
img = img / 255.0  # Normalize
cv.imshow("Patch", img)
cv.waitKey(0)
cv.destroyAllWindows()

