import shutil
import cv2
import os
import csv

import processors.dataset_A_processor as processor
import processors.dataset_A_augmentation as augmentation

########################### PARAMETERS ###########################

# Path to the CAPTCHA samples directory
SAMPLES_DIR = 'samples/type_A'

# Path to the dataset directory (where the images will be saved)
DATASET_DIR = 'datasets/datasetA'

# Path to the csv file containing the labels
CSV_PATH = 'datasets/dataset_A.csv'

# if DATASET_DIR or CSV_PATH already exist, they will be overwritten



####################### DATASET GENERATION #######################

if __name__ == '__main__':
    # For every image in the samples directory get the bounding boxes and create the dataset
    X = []
    y2 = []
    #j = []

    # Start populating the dataset
    for photo in os.listdir(SAMPLES_DIR):
        if photo.endswith('.png') or photo.endswith('.jpg'):

            crop_img = processor.process_image(os.path.join(SAMPLES_DIR, photo))
            filename = photo[:-4]

            for i in range(5):
                X.append(crop_img[i])
                y2.append(filename[i])
                #j.append(filename + str(i))

    # Let's save the dataset in a convenient folder structure

    # Remove the dataset if it already exists
    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)

    os.makedirs(DATASET_DIR)

    # Initialize the csv file
    with open(CSV_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'label'])

    # Save the images in the folders
    for i, img in enumerate(X):
        # Resize the image to 50x50
        img = cv2.resize(img, (50, 50))
        # Make the image binary
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        # Save the image in the dataset folder
        cv2.imwrite(os.path.join(DATASET_DIR, str(i) + '.png'), img)
        # Add a line to the csv file with the image name and the label
        with open(CSV_PATH, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
            # Create the map from the label to the letter
            map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35}
            # Write the line in the csv file
            writer.writerow([i, map[y2[i]]])

    # Augment the data
    augmentation.augment_data(CSV_PATH, DATASET_DIR)
