# Create a progrogram that takes in a dataset and augments it by rotating the images between -45 and 45 degrees.
#

# import the necessary packages
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import random
import numpy as np
import csv


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=255)
    return result


# Read the csv file
df = pd.read_csv('datasets/dataset_1.csv')

# Get the number of images in the dataset
num_images = len(df)

# Iterate in the dataset folder
for image in os.listdir("./datasets/dataset/"):
    # Get image name
    image_name = image.split('.')[0]
    # Get image label

    image_label = df.iloc[int(image_name), 1]
    # Read the image in grayscale
    image = cv2.imread("./datasets/dataset/" + image, 0)
    # Add a white background to the image
    image = cv2.copyMakeBorder(
        image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # Generate a random integer number between -45 and 45
    random_number = random.randint(-30, 30)
    # Rotate the image
    rotated_image = rotate_image(image, random_number)
    # Resize the image
    rotated_image = cv2.resize(rotated_image, (50, 50))
    num_images += 1
    # Save the image
    cv2.imwrite("./datasets/dataset/" + str(num_images) + '.png', rotated_image)
    # Write the image name and label to the csv file
    with open('datasets/dataset_1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
        # Create the map from the label to the letter
        map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35}
        # Write the line in the csv file
        # Show the image name and the label
        writer.writerow([num_images, image_label])
