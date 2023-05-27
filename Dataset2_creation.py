import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv


files_path = "./samples_2/"

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


count = 0

with open('dataset_2.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'label'])

# Read all files in folder
for image in os.listdir(files_path):
    # Read image
    img = cv2.imread(files_path + image)

    # Apply transformation
    img = transform_image(img)

    # Get rectangles
    rects = split(img)

    # Save each letter in a folder
    for i, rect in enumerate(rects):
        if (len(rects)>10):
            # Ignore the image
            break
        count += 1
        # Save the image in the folder in grayscale
        cv2.imwrite('./datasetN2/' + str(count) + '.png', img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) 
        # Save a csv file with the name of the new image and the letter it contains (for training)
        with open('dataset_2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
            # Create the map from the label to the letter
            map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35, 'A':36, 'B':37, 'C':38, 'D':39, 'E':40, 'F':41, 'G':42, 'H':43, 'I':44, 'J':45, 'K':46, 'L':47, 'M':48, 'N':49, 'O':50, 'P':51, 'Q':52, 'R':53, 'S':54, 'T':55, 'U':56, 'V':57, 'W':58, 'X':59, 'Y':60, 'Z':61}
            # Write the line in the csv file
            writer.writerow([count, map[image[i]]])


# Cicle through all the images in the folder and resize them to 28x28
for image in os.listdir('./datasetN2/'):
    img = cv2.imread('./datasetN2/' + image)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite('./datasetN2/' + image, img)
