import cv2
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser('preprocess image')
parser.add_argument('--input_dir', help='input directory for images want to preprocess.', type=str, default='./Training dataset/label_img')
parser.add_argument('--output_dir', help='output directory for images after preprocessing.', type=str, default='./Training dataset/test_fold')
args = parser.parse_args()

for filename in os.listdir(args.input_dir):
    input_path = os.path.join(args.input_dir, filename)
    output_path = os.path.join(args.output_dir, filename)
    
    image = cv2.imread(input_path)

    image1 = image.copy()
    image2 = image.copy()


    for row in range(image1.shape[0]):
        encounter_white = False
        for col in range(image1.shape[1]):
            if image1[row, col, 0] != 0 and not encounter_white:
                encounter_white = True;

            if not encounter_white:
                continue
            else:
                image1[row, col, :] = [255, 0, 255]

    for row in range(image2.shape[0]-1, 0, -1):
        encounter_white = False
        for col in range(image2.shape[1]-1, 0, -1):
            if image2[row, col, 0] != 0 and not encounter_white:
                encounter_white = True;

            if not encounter_white:
                continue
            else:
                image2[row, col, :] = [255, 0, 255]
                
    ### store image
    print(filename)
    new_img = cv2.bitwise_and(image1, image2)
    new_img[:, :, 1] = np.bitwise_or(new_img[:, :, 1], image[:, :, 1])
    cv2.imwrite(output_path, new_img)