from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

mser = cv2.MSER_create()

path = 'samples/'

def get_bounding_boxes(img, X, y2, filename, j):

    cv2.rectangle(img, (30,12), (50,49), 0, 1)
    cv2.rectangle(img, (50,12), (70,49), 0, 1)
    cv2.rectangle(img, (70,12), (90,49), 0, 1)
    cv2.rectangle(img, (90,12), (110,49),0, 1)
    cv2.rectangle(img, (110,12),(130,49),0, 1)

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

        # Preprocessing
        filename = photo[:-4]
        img1_thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        # remove useless stuff
        img1_thr[:, 0:25] = 255
        img1_thr[:, 160:] = 255

        smooth = cv2.GaussianBlur(img1_thr, (3,3), 0)

        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(smooth, -1, sharpen_kernel)

        ret, thresh = cv2.threshold(sharpen, 185, 255, cv2.THRESH_BINARY)


        get_bounding_boxes(thresh, X, y2, filename, j)

# Let's save the dataset in a convenient folder structure

# Create the dataset folder
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Create the folders for each letter
for letter in 'abcdefghijklmnopqrstuvwxyz0123456789':
    if not os.path.exists('dataset/' + letter):
        os.makedirs('dataset/' + letter)

# Save the images in the folders
for i, img in enumerate(X):
    # Resize the image to 50x50
    img = cv2.resize(img, (50, 50))
    # Make the image binary
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Save the image
    cv2.imwrite('dataset/' + y2[i] + '/' + str(i) + '.png', img)

X = np.array(X, dtype=np.float32)
y2 = np.array(y2)

print(X.shape)
print(y2.shape)

