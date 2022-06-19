"""
GENERATE DATA FOR NUMBERS RECOGNITION
"""

import cv2
import os
import numpy as np
from sudoku_segmentation_prediction import multipleExtractNumbersImgFromSudokuImg



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def extractNumbersFromMultipleSudokuGrid(sudokus):
      numbers = []
      numbers.append(multipleExtractNumbersImgFromSudokuImg(sudokus))
      numbers = np.array(numbers)
      numbers = numbers.reshape(numbers.shape[0]*numbers.shape[1]*numbers.shape[2],numbers.shape[3],numbers.shape[4],numbers.shape[5])
      return numbers
      

images = load_images_from_folder("sudoku_img/detection_train")
images = np.array(images)
print(images.shape)

imagesToSave = extractNumbersFromMultipleSudokuGrid(images)
print(imagesToSave.shape)

savedPath = "../sudoku_img/numbers_training"

print("BEGIN")
for count, image in enumerate(imagesToSave):
      cv2.imwrite(os.path.join(savedPath,f'{count}.jpg'), image)
      if count % 1000 == 0:
            print("image number ",count)
print("END")


