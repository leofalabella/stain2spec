from openslide import OpenSlide
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import pca
import numpy as np

img = np.load('/home/leo-f/stain2spec/data/processed/HE/test/00000_test_1+.npy')
print(img.shape)