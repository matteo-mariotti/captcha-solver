import cv2
import numpy as np


def split(image):
    mser = cv2.MSER_create()

    _, rects = mser.detectRegions(image)

    # sort rectangles by left pixel
    rects = sorted(rects, key=lambda x: x[0])
    return rects


def transform_image(img):
    # make all gray pixel brighter than [200, 200, 200] white (to remove the background noise)
    img[np.where((img > [200, 200, 200]).all(axis=2))] = [255, 255, 255]

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur the image (to remove airtfacts)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # sharpen the image (to make the letters clearer)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blurred, -1, sharpen_kernel)

    # threshold image (to convert it to a binary image)
    _, thresh = cv2.threshold(sharpen, 220, 255, cv2.THRESH_BINARY)

    return thresh


def process_image(path):
    X = []
    # Read image
    img = cv2.imread(path)

    # Apply transformation
    img = transform_image(img)

    # Get rectangles
    rects = split(img)

    for rect in rects: X.append(cv2.resize(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], (28, 28)))

    return X