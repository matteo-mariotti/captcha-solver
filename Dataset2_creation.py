import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


files_path = "../samples_2/"

def split(image):
    mser = cv2.MSER_create()

    regions, rects = mser.detectRegions(image)

    # sort rectangles by left pixel
    rects = sorted(rects, key=lambda x: x[0])

    return rects


def transform_image(img):

    # make all gray pixel above 200 white
    img[np.where((img > [200, 200, 200]).all(axis=2))] = [255, 255, 255]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blurred, -1, sharpen_kernel)

    # threshold image
    ret, thresh = cv2.threshold(sharpen, 220, 255, cv2.THRESH_BINARY)

    return thresh

# Read all files in folder
for image in os.listdir(files_path):

    img = cv2.imread(image)

    # Apply transformation
    img = transform_image(img)

    # Get rectangles
    rects = split(img)

    # subplot every rectangle
    for i, rect in enumerate(rects):
        plt.imshow(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], cmap='gray')

