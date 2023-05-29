import shutil
from PIL import Image
import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

mser = cv2.MSER_create()

path = 'samples/'

def get_bounding_boxes(img, X, y2, filename, j):

    # Let's crop each letter
    crop = img[12:49, 30:50]
    X.append(crop)
    y2.append(filename[0])
    j.append(filename + str(0))

    crop = img[12:49, 50:70]
    X.append(crop)
    y2.append(filename[1])
    j.append(filename + str(1))

    crop = img[12:49, 70:90]
    X.append(crop)
    y2.append(filename[2])
    j.append(filename + str(2))

    crop = img[12:49, 90:110]
    X.append(crop)
    y2.append(filename[3])
    j.append(filename + str(3))

    crop = img[12:49, 110:130]
    X.append(crop)
    y2.append(filename[4])
    j.append(filename + str(4))

# For every image in the samples directory get the bounding boxes and create the dataset
X = []
y2 = []
j = []

for photo in os.listdir(path):
    if photo.endswith('.png') or photo.endswith('.jpg'):
        img = cv2.imread(path + photo, cv2.IMREAD_GRAYSCALE)

        #pre_smooth = cv2.GaussianBlur(img, (3,3), 0)

        # Preprocessing
        filename = photo[:-4]
        img1_thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        # remove useless stuff
        img1_thr[:, 0:25] = 255
        img1_thr[:, 160:] = 255

        #close_img1 = cv2.morphologyEx(img1_thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        smooth = cv2.GaussianBlur(img1_thr, (3,3), 0)

        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(smooth, -1, sharpen_kernel)

        ret, thresh = cv2.threshold(sharpen, 185, 255, cv2.THRESH_BINARY)


        get_bounding_boxes(thresh, X, y2, filename, j)

# Let's save the dataset in a convenient folder structure

# Create the dataset folder
if os.path.exists('datasets/dataset'):
    shutil.rmtree('datasets/dataset')

os.makedirs('datasets/dataset')

if os.path.exists('datasets/dataset_1.csv'):
    os.remove('datasets/dataset_1.csv')

    with open('datasets/dataset_1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'label'])


# Save the images in the folders
for i, img in enumerate(X):
    # Resize the image to 50x50
    img = cv2.resize(img, (50, 50))
    # Make the image binary
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Save the image in the dataset folder
    cv2.imwrite('datasets/dataset/' + str(i) + '.png', img)
    # Add a line to the csv file with the image name and the label
    with open('datasets/dataset_1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
        # Create the map from the label to the letter
        map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35}
        # Write the line in the csv file
        writer.writerow([i, map[y2[i]]])
    
