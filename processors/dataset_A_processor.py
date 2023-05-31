import cv2
import numpy as np

def get_bounding_boxesV2(img):

    X = []

    mser = cv2.MSER_create()

    regions, rects = mser.detectRegions(img)

    im2 = img.copy()
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
    final_rects = [rect for rect in final_rects if np.sum(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) < 254 * rect[2] * rect[3]]

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

        crop = im2[12:49, 50:70]
        X.append(crop)

        crop = im2[12:49, 70:90]
        X.append(crop)

        crop = im2[12:49, 90:110]
        X.append(crop)

        crop = im2[12:49, 110:130]
        X.append(crop)
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
    
    return X

def process_image(path):

    # Read the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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
    return get_bounding_boxesV2(thresh)