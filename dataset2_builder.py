import shutil
import cv2
import os
import csv

import processors.dataset2_processor as processor


########################### PARAMETERS ###########################

# Path to the CAPTCHA samples directory
SAMPLES_DIR = 'samples/type_2'

# Path to the dataset directory (where the images will be saved)
DATASET_DIR = 'datasets/datasetN2'

# Path to the csv file containing the labels
CSV_PATH = 'datasets/dataset_2.csv'

# if DATASET_DIR or CSV_PATH already exist, they will be overwritten


####################### DATASET GENERATION #######################

if __name__ == '__main__':
    # Delete the dataset folder if it already exists
    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)
    
    os.makedirs(DATASET_DIR)
    
    # Initialize the csv file
    with open(CSV_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'label'])
    
    
    # Start populating the dataset
    count = 0
    for image in os.listdir(SAMPLES_DIR):
    
        X = processor.process_image(os.path.join(SAMPLES_DIR, image))    
    
        # Save each letter in a folder
        for i, rect in enumerate(X):
            if (len(X) != 10):
                # Ignore the image
                break
            # Save the image in the folder in grayscale
            cv2.imwrite(os.path.join(DATASET_DIR, str(count) + '.png'), X[i]) 
            # Save a csv file with the name of the new image and the letter it contains (for training)
            with open(CSV_PATH, 'a', newline='') as file:
                writer = csv.writer(file)
                # Write the image name and the label in the csv file, the label should be bteewen 0 and 35 (26 letters + 10 numbers), 0->0, 1->1, ..., 9->9, A->10, B->11, ..., Z->35
                # Create the map from the label to the letter
                map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35, 'A':36, 'B':37, 'C':38, 'D':39, 'E':40, 'F':41, 'G':42, 'H':43, 'I':44, 'J':45, 'K':46, 'L':47, 'M':48, 'N':49, 'O':50, 'P':51, 'Q':52, 'R':53, 'S':54, 'T':55, 'U':56, 'V':57, 'W':58, 'X':59, 'Y':60, 'Z':61}
                # Write the line in the csv file
                writer.writerow([count, map[image[i]]])
            count += 1