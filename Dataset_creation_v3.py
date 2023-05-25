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

def get_bounding_boxesV2(thresh, X, y2, filename, j):

    mser = cv2.MSER_create()

    regions, rects = mser.detectRegions(thresh)

    im2 = thresh.copy()
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)

    final_rects = []

    for (x, y, w, h) in rects:
        if w > 135: continue
        elif w > 110:
            #print("splitting in 5")
            final_rects.append((x, y, w//5, h))
            final_rects.append((x + w//5, y, w//5, h))
            final_rects.append((x + 2 * w//5, y, w//5, h))
            final_rects.append((x + 3 * w//5, y, w//5, h))
            final_rects.append((x + 4 * w//5, y, w//5, h))
        elif w > 80:
            #print("splitting in 4")
            final_rects.append((x, y, w//4, h))
            final_rects.append((x + w//4, y, w//4, h))
            final_rects.append((x + 2 * w//4, y, w//4, h))
            final_rects.append((x + 3 * w//4, y, w//4, h))
        elif w > 50:
            #print("splitting in 3")
            final_rects.append((x, y, w//3, h))
            final_rects.append((x + w//3, y, w//3, h))
            final_rects.append((x + 2 * w//3, y, w//3, h))

        elif w > 35:
            #print("splitting in 2")
            final_rects.append((x, y, w//2, h))
            final_rects.append((x + w//2, y, w//2, h))
        else:
            final_rects.append((x, y, w, h))

    # remove rectangles that contain basically only white pixels
    final_rects = [rect for rect in final_rects if np.sum(thresh[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) < 254 * rect[2] * rect[3]]

    count = 0


    # remove rectangles with a height of less than 20 pixels
    #final_rects = [rect for rect in final_rects if rect[3] > 20]

    # merge regions that are horizontally contained (take the area from the top left to the bottom right)
    final_rects = [rect for rect in final_rects if not any([rect[0] > other_rect[0] and rect[1] > other_rect[1] and rect[0] + rect[2] < other_rect[0] + other_rect[2] and rect[1] + rect[3] < other_rect[1] + other_rect[3] for other_rect in final_rects if other_rect != rect])]


    # Sort the rectangles by their x coordinate
    final_rects = sorted(final_rects, key=lambda x: x[0])

    count = 0

    while count < 4 and len(final_rects) > 5:
        # check if the first two rectangles overlap
        if final_rects[count][0] + final_rects[count][2] > final_rects[count + 1][0]:

            #print(final_rects[count + 1][0] - final_rects[count][0])

            #print("count is " + str(count))

            x1 = final_rects[count][0]
            w1 = final_rects[count][2]
            x2 = final_rects[count + 1][0]
            w2 = final_rects[count + 1][2]

            if (final_rects[count + 1][0] - final_rects[count][0] < 10) or ((x2 + w2) < (x1 + w1) + 5):
                #print("Overlap detected")
                # if they do, merge them

                new_x = final_rects[count][0]
                new_y = min(final_rects[count][1], final_rects[count + 1][1])
                new_w = max(final_rects[count][0] + final_rects[count][2], final_rects[count + 1][0] + final_rects[count + 1][2]) - final_rects[count][0]
                new_h = max(final_rects[count][1] + final_rects[count][3], final_rects[count + 1][1] + final_rects[count + 1][3]) - min(final_rects[count][1], final_rects[count + 1][1])

                final_rects[count] = (new_x, new_y, new_w, new_h)
                # and remove the second one
                final_rects.pop(count + 1)
                continue

        count += 1

    # remove rectangles with a height of less than 20 pixels
    final_rects = [rect for rect in final_rects if (rect[3] > 20 and rect[2] > 10)]

    if len(final_rects) != 5: 
        # Split using the fixed positions
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
    else:
        # take the 5 biggest regions
        #final_rects = sorted(final_rects, key=lambda x: x[2] * x[3], reverse=True)[:5]
        # this is for safety but it should be improved


        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0), (255,0,255)]

        # Draw the rectangles  
        for i, rect in enumerate(final_rects):
            print(i)
            cv2.rectangle(im2, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), colors[i], 1) 
            
        for i, (x, y, w, h) in enumerate(final_rects):
            #cv2.rectangle(im2, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
            crop = im2[y:y+h, x:x+w]
            # Resize to 50x50
            crop = cv2.resize(crop, (50, 50))
            X.append(crop)
            y2.append(filename[i])

    #plt.imshow(im2)

    # save the image
    #cv2.imwrite('test_results/' + filename + '.png', im2)
    #print("saved image as " + filename + ".png")


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


        get_bounding_boxesV2(thresh, X, y2, filename, j)

# Let's save the dataset in a convenient folder structure

# Create the dataset folder
if not os.path.exists('datasetV2'):
    os.makedirs('datasetV2')

# Save the images in the folders
for i, img in enumerate(X):
    # Resize the image to 50x50
    img = cv2.resize(img, (50, 50))
    # Make the image binary
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Save the image in the dataset folder
    cv2.imwrite('datasetV2/' + str(j[i]) + str(i) + '.png', img)
    # Add a line to the csv file with the image name and the label
    with open('dataset_2.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
        # Create the map from the label to the letter
        map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35}
        # Write the line in the csv file
        writer.writerow([i, map[y2[i]]])
    
