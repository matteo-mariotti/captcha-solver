import torch
import torchvision.transforms as transforms
import cv2

import processors.dataset_A_processor as dataset_A_processor
import processors.dataset_B_processor as dataset_B_processor

MODEL1_PATH = 'models/model_A.pt'
MODEL2_PATH = 'models/model_B.pt'

CAPTCHA_PATH = 'samples/type_A/728n8.png'
#CAPTCHA_PATH = 'samples/type_2/0a2GPKF628.jpg'

PLATFORM = "cuda" if torch.cuda.is_available() else "cpu"

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


if __name__ == '__main__':

    print('Processing file: ' + CAPTCHA_PATH)

    # we need to detect if the CAPTCHA is from the first or the second set
    # we can do this by checking the image shape

    img = cv2.imread(CAPTCHA_PATH)
    if img.shape[0] == 50 and img.shape[1] == 200:
        # image is from the first set
        print('image is from SET 1', "splitting the image in characters...", sep="\n")
        # split the captcha in characters using the correct algorithm
        X = dataset_A_processor.process_image(CAPTCHA_PATH)

        # load the model
        print("loading the model...")
        model = torch.jit.load(MODEL1_PATH)
        model = model.to(PLATFORM)
        model.eval()
    else:
        # image is from the second set
        print('image is from SET 2', "splitting the image in characters...", sep="\n")
        # split the captcha in characters using the correct algorithm
        X = dataset_B_processor.process_image(CAPTCHA_PATH)

        # load the model
        print("loading the model...")
        model = torch.jit.load(MODEL2_PATH)
        model = model.to(PLATFORM)
        model.eval()

    # convert the images to PyTorch tensors
    image_tensors = []
    for i in range(len(X)):
        image_tensors.append(transforms.ToTensor()(X[i]))

    # stack the tensors in a single tensor
    images = torch.stack(image_tensors)

    # now we have a tensor with all the characters of the captcha
    # we can use the model to predict the characters

    # do the prediction
    with torch.no_grad():
        print("predicting (using", PLATFORM, "platform)...")
        images = images.to(PLATFORM)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # match the predicted labels with the corresponding characters
        predicted_letters = (CHARACTERS[predicted[i]] for i in range(len(predicted)))

        # print the predicted captcha value
        print("Predicted value:", "".join(predicted_letters))
