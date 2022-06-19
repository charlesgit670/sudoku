
"""
NUMBERS RECOGNITION PREDICTION
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import shutil
import os

import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
from PIL import Image

def model():
      '''
      Load model and weight

      Returns model
      '''
      model = models.load_model(os.path.abspath(os.path.dirname( __file__ ))+"/../model_training/output/model_numbersReading")
      model.load_weights(os.path.abspath(os.path.dirname( __file__ ))+"/../model_training/output/model_numbersReading_weights")
      return model

def convertNUmbersImgToArray(im):
      '''
      Predict 81 images for each cell of the sudoku 
      
      Arguments:
            im: All image of number in a sudoku grid (81,28,28,1)
            model: model use for prediction
      Returns
            result: array (9,9) with values extract from image
      '''
      
      row = im.shape[0]
      column = im.shape[1]

      sudokuNumbersImgReSizeAndGrayScale = []
      for i in range(row):
            for j in range(column):
                  imgTemp = cv2.cvtColor(im[i][j] , cv2.COLOR_BGR2GRAY)
                  imgTemp = cv2.resize(imgTemp, (28,28)).reshape(28,28,1)
                  sudokuNumbersImgReSizeAndGrayScale.append(imgTemp)
      sudokuNumbersImgReSizeAndGrayScale = np.array(sudokuNumbersImgReSizeAndGrayScale)

      sudokuNumbersImgReSizeAndGrayScale = sudokuNumbersImgReSizeAndGrayScale / 255
      sudokuNumbersImgReSizeAndGrayScale = abs(sudokuNumbersImgReSizeAndGrayScale - 1)

      output = model().predict(sudokuNumbersImgReSizeAndGrayScale)
      result = []
      for i in output:
            result.append(np.argmax(i))
      result = np.array(result).reshape(9,9, order='F')
      return result
      
def convertNUmbersImgToArrayTest(im):
      '''
      a test function to visualize image with correspondant prediction
      '''
      modelRecognition = model()
      for i in im:
            output = modelRecognition.predict(i.reshape(1,28,28,1))
            print(output, ' value : ',np.argmax(output))
            cv2.imshow('image',i)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
      return output


def sortImgNumber():
      '''
      Function used to sort number with model instead of sorting by hand (not 100% accurate)
      '''
      files = glob.glob('sudoku_img/numbers_training/*.jpg')
      modelRecognition = model()
      for file in files:           
            x_train = np.array(Image.open(file).convert('L'))
            x_train = x_train / 255
            x_train = abs(x_train - 1)
            x_train = x_train.reshape(-1,28,28,1)
            output = modelRecognition.predict(x_train)
            number = np.argmax(output)
      
            shutil.copy(file, 'sudoku_img/'+str(number)+'bis/')
                                                                    









