"""
MAIN SCRIPT

"""
import cv2
import copy

from model_prediction.sudoku_segmentation_prediction import extractNumbersImgFromSudokuImg
from model_prediction.numbers_recognition_prediction import convertNUmbersImgToArray, convertNUmbersImgToArrayTest
from model_prediction.sudoku_solver_prediction import sudoku_solver

#path of the sudoku image you want to solve
im = cv2.imread("sudoku_img/detection_test/image47.jpg")

sudokuNumbersImg = extractNumbersImgFromSudokuImg(im)
sudokuGrid = convertNUmbersImgToArray(sudokuNumbersImg)

sudokuUnsolved = copy.copy(sudokuGrid)
sudokuSolved = sudoku_solver(sudokuGrid)

print(sudokuUnsolved)
print(sudokuSolved)

cv2.imshow('image',im)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# test = convertNUmbersImgToArrayTest(sudokuNumbersImg)
# print(test)