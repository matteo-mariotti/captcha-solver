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

def get_bounding_boxesV2(thresh, X, y2, filename, j):

    mser = cv2.MSER_create()

    regions, rects = mser.detectRegions(thresh)

    im2 = thresh.copy()
    final_rects = []

    # split rectangles that are too wide
    # because they will contain more than one character
    for (x, y, w, h) in rects:
        if w > 135: continue # if larger than 135, we ignore it
        elif w > 100:
            # splitting the rectangle in 5 subrectangles
            final_rects.append((x, y, w//5, h))
            final_rects.append((x + w//5, y, w//5, h))
            final_rects.append((x + 2 * w//5, y, w//5, h))
            final_rects.append((x + 3 * w//5, y, w//5, h))
            final_rects.append((x + 4 * w//5, y, w//5, h))
        elif w > 80:
            # splitting the rectangle in 4 subrectangles
            final_rects.append((x, y, w//4, h))
            final_rects.append((x + w//4, y, w//4, h))
            final_rects.append((x + 2 * w//4, y, w//4, h))
            final_rects.append((x + 3 * w//4, y, w//4, h))
        elif w > 50:
            # splitting the rectangle in 3 subrectangles
            final_rects.append((x, y, w//3, h))
            final_rects.append((x + w//3, y, w//3, h))
            final_rects.append((x + 2 * w//3, y, w//3, h))

        elif w > 35:
            # splitting the rectangle in 2 subrectangles
            final_rects.append((x, y, w//2, h))
            final_rects.append((x + w//2, y, w//2, h))
        else:
            # rectangle is small enough, it probably contains only one character
            final_rects.append((x, y, w, h))

    # remove rectangles that contain basically only white pixels
    final_rects = [rect for rect in final_rects if np.sum(thresh[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) < 254 * rect[2] * rect[3]]

    # merge regions that are horizontally contained (take the area from the top left to the bottom right)
    final_rects = [rect for rect in final_rects if not any([rect[0] > other_rect[0] and rect[1] > other_rect[1] and rect[0] + rect[2] < other_rect[0] + other_rect[2] and rect[1] + rect[3] < other_rect[1] + other_rect[3] for other_rect in final_rects if other_rect != rect])]

    # Sort the rectangles by their top left X coordinate
    final_rects = sorted(final_rects, key=lambda x: x[0])


    # merge rectangles that overlap
    count = 0
    while count < 4 and len(final_rects) > 5:
        # check if the first two rectangles overlap
        if final_rects[count][0] + final_rects[count][2] > final_rects[count + 1][0]:

            x1 = final_rects[count][0]
            w1 = final_rects[count][2]
            x2 = final_rects[count + 1][0]
            w2 = final_rects[count + 1][2]

            # if the rectangles X coordinates are very close, we merge them
            # (note that this is based on the fact that rectangles are ordered by their top left X coordinate)
            if (final_rects[count + 1][0] - final_rects[count][0] < 10) or ((x2 + w2) < (x1 + w1) + 5):

                # define the new rectangle
                new_x = final_rects[count][0]
                new_y = min(final_rects[count][1], final_rects[count + 1][1])
                new_w = max(final_rects[count][0] + final_rects[count][2], final_rects[count + 1][0] + final_rects[count + 1][2]) - final_rects[count][0]
                new_h = max(final_rects[count][1] + final_rects[count][3], final_rects[count + 1][1] + final_rects[count + 1][3]) - min(final_rects[count][1], final_rects[count + 1][1])

                # replace the first old rectangle with the new one
                final_rects[count] = (new_x, new_y, new_w, new_h)

                # remove the second old rectangle
                final_rects.pop(count + 1)
                continue

        # increment the counter
        # the counter is only incremented if no merging happened
        # to avoid looping forever
        count += 1

    # remove rectangles with a height of less than 20 pixels or a width of less than 10 pixels
    # they are just noise, no letter can be that small
    final_rects = [rect for rect in final_rects if (rect[3] > 20 and rect[2] > 10)]

    # if we do not have 5 rectangles, the algorithm failed
    # so we try to split the image using fixed positions
    if len(final_rects) != 5: 
        # Split using the fixed positions
        # Let's crop each letter
        crop = im2[12:49, 30:50]
        X.append(crop)
        y2.append(filename[0])
        j.append(filename + str(0))

        crop = im2[12:49, 50:70]
        X.append(crop)
        y2.append(filename[1])
        j.append(filename + str(1))

        crop = im2[12:49, 70:90]
        X.append(crop)
        y2.append(filename[2])
        j.append(filename + str(2))

        crop = im2[12:49, 90:110]
        X.append(crop)
        y2.append(filename[3])
        j.append(filename + str(3))

        crop = im2[12:49, 110:130]
        X.append(crop)
        y2.append(filename[4])
        j.append(filename + str(4))
    else:
        for i, (x, y, w, h) in enumerate(final_rects):
            # Crop the image
            crop = im2[y:y+h, x:x+w]
            # Resize to 50x50
            crop = cv2.resize(crop, (50, 50))

            # blur the cropped image
            crop = cv2.GaussianBlur(crop, (9, 9), 0)

            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            crop = cv2.filter2D(crop, -1, sharpen_kernel)

            _, crop = cv2.threshold(crop, 185, 255, cv2.THRESH_BINARY)       

            X.append(crop)
            y2.append(filename[i])


# For every image in the samples directory get the bounding boxes and create the dataset
X = []
y2 = []
j = []

# Start populating the dataset
for photo in os.listdir(path):
    if photo.endswith('.png') or photo.endswith('.jpg'):

        # Read the image
        img = cv2.imread(path + photo, cv2.IMREAD_GRAYSCALE)
        filename = photo[:-4]

        # Process the image

        # threshold the image to remove the background
        img1_thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        # remove parts of the image that contain only noise
        # as there are no letters close to the edges of the image
        img1_thr[:, 0:25] = 255
        img1_thr[:, 160:] = 255

        # do a morphological closing to close the gaps between letters
        close_img1 = cv2.morphologyEx(img1_thr, cv2.MORPH_CLOSE, np.ones((3,1), np.uint8))

        # blur the image to smooth it and remove noise
        smooth = cv2.GaussianBlur(close_img1, (3,3), 0)

        # sharpen the image to make the letters more clear
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(smooth, -1, sharpen_kernel)

        # threshold the image to get a binary image again
        ret, thresh = cv2.threshold(sharpen, 185, 255, cv2.THRESH_BINARY)

        # compute the bounding boxes around the letters
        get_bounding_boxesV2(thresh, X, y2, filename, j)


# Let's save the dataset in a convenient folder structure

# Remove the dataset if it already exists
if os.path.exists('datasets/dataset'): shutil.rmtree('datasets/dataset')
if os.path.exists('datasets/dataset_1.csv'): os.remove('datasets/dataset_1.csv')

os.makedirs('datasets/dataset')

# Initialize the csv file
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
