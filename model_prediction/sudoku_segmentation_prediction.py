"""
SUDOKU SEGMENTATION PREDICTION
"""
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

def model():
      '''
      load segmentation model with weight
      
      Returns model
      '''
      print()
      cfg = get_cfg()
      cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
      cfg.MODEL.WEIGHTS = os.path.abspath(os.path.dirname( __file__ ))+"/../model_training/output/model_final.pth"
      cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
      
      return DefaultPredictor(cfg)
      
def retrieveSudokuGridImg(image, predictor):
      '''
      Extract sudoku grid

      Arguments:
            image: sudoku image
            predictor: model predictor
      Returns:
            imCropped: sudoku grid extract
      '''
      outputs = predictor(image)
      index = 0
      # if more than 2 grid is detected -> retreive the one with larger area
      if outputs["instances"].to("cpu").pred_classes.numpy().size > 1:
            f = lambda array: (array[:,2]-array[:,0])*(array[:,3]-array[:,1])
            area = f(outputs["instances"].to("cpu").pred_boxes.tensor.numpy())
            index = np.argmax(area)            
      assert outputs["instances"].to("cpu").pred_classes.numpy()[index] == 0, 'no sudoku grid detected'
      mask = outputs["instances"].to("cpu").pred_masks.numpy()[index]
      segm = np.argwhere(mask==True)
      corners = findCorner(segm)
      index = []
      for corner in corners:
            index.append(findShortestPath(corner,segm))
      imCropped = warpImage(image, index, corners)
      return imCropped

def findShortestPath(point, arrays):
      '''
      Find coordinates in arrays closer to this point

      Arguments:
            point: coord (x,y)
            arrays: sudoku segmentation arrays 
      Returns:
            coordinate of the segmentation arrays closer to the point
      '''
      f = lambda array: (point[0]-array[:,0])**2 + (point[1]-array[:,1])**2
      lenghtArray = f(arrays)
      index = np.argmin(lenghtArray)
      return arrays[index]

def findCorner(segm):
     '''
      Find the 4 closest points that frame the segmentation image

      Arguments:
            segm: segmentation image
      Returns:
            coordinates of the 4 points
      ''' 
     yMin = segm[:,0].min()
     xMin = segm[:,1].min()
     yMax = segm[:,0].max()
     xMax = segm[:,1].max()
     
     return np.array([[yMin,xMin],[yMin,xMax],[yMax,xMin],[yMax,xMax]])

def warpImage(image, input_pts, output_pts):
      '''
      Warp image

      Arguments:
            image: sudoku image
            input_pts: corners of the sudoku grid 
            output_pts: corners of the output image
      Returns:
            image warped
      '''
      input_pts = np.flip(input_pts,1)
      height = abs(output_pts[0,1] - output_pts[3,1])
      width = abs(output_pts[0,0] - output_pts[3,0])
      M = cv2.getPerspectiveTransform(np.array(input_pts,np.float32),np.array([[0,0],[width,0],[0,height],[width,height]],np.float32))
      out = cv2.warpPerspective(image,M,(width,height))
      return out
                              
def splitGridImg(image):
      '''
      split the sudoku grid into 9x9 images 

      Arguments:
            image: sudoku grid image
      Returns:
            9x9 cells
      '''
      x = image.shape[0]
      y = image.shape[1]
      newX = int(x/9)*9
      newY = int(y/9)*9
      resizeImg = np.array(cv2.resize(image, (newX, newY)))
      split1 = np.array(np.split(resizeImg, 9, axis=0))
      split2 = np.array(np.split(split1, 9, axis=2)) 
      
      return np.array(split2)

def extractNumbersImgFromSudokuImg(image):
      '''
      Extract image of each number in the sudoku grid

      Arguments:
           image: sudoku image
      Returns:
            Arrays of (9,9,height,width,channel)
      '''
      imCropped = retrieveSudokuGridImg(image, model())  
      
      return splitGridImg(imCropped)

#######################################################################################################################
#########################   FUNCTIUN USED TO GENERATE DATA  FROM REAL CASE  ###########################################
#######################################################################################################################

def multipleRetrieveSudokuGridImg(images, predictor):
      imCropped = []
      for image in images:
           index = 0
           outputs = predictor(image)
           if outputs["instances"].to("cpu").pred_classes.numpy().size > 1:
                 f = lambda array: (array[:,2]-array[:,0])*(array[:,3]-array[:,1])
                 area = f(outputs["instances"].to("cpu").pred_boxes.tensor.numpy())
                 index = np.argmax(area)            
           assert outputs["instances"].to("cpu").pred_classes.numpy()[index] == 0, 'no sudoku grid detected'
           mask = outputs["instances"].to("cpu").pred_masks.numpy()[index]
           segm = np.argwhere(mask==True)
           corners = findCorner(segm)
           index = []
           for corner in corners:
                 index.append(findShortestPath(corner,segm))
           imCropped.append(warpImage(image, index, corners))
                 
      return imCropped

def multipleSplitGridImg(images):
      gridSplit = []
      for image in images:
          x = image.shape[0]
          y = image.shape[1]
          newX = int(x/9)*9
          newY = int(y/9)*9
          resizeImg = np.array(cv2.resize(image, (newX, newY)))
          split1 = np.array(np.split(resizeImg, 9, axis=0))
          split2 = np.array(np.split(split1, 9, axis=2))
          gridSplit.append(resizeMatrixImages(split2, 28, 28))        
      return np.array(gridSplit)

def resizeMatrixImages(images, width, height):
      imagesResized = []
      for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                  imagesResized.append(cv2.resize(images[i][j], (width,height)))
      return np.array(imagesResized)  

def multipleExtractNumbersImgFromSudokuImg(images):
      '''
      Extract each cell of sudoku images

      Arguments:
            images: sudoku images
      Returns:
            each cell from all sudoku images
      '''
      imCropped = multipleRetrieveSudokuGridImg(images, model())     
      return multipleSplitGridImg(imCropped)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#display sudoku image with segmentation

# im = cv2.imread("../sudoku_img/detection_test/image175.jpg")
# model = model()
# outputs = model(im)

# v = Visualizer(im[:, :, ::-1],
#     scale=1
# )
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('image_show',out.get_image()[:, :, ::-1])
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# display all sudoku cell extracted

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(10, 7))
# im = cv2.imread("../sudoku_img/detection_test/image175.jpg")
# outputs = extractNumbersImgFromSudokuImg(im)
                 
# fig = plt.figure(figsize=(10, 7))

# count = 1
# for x in range(9):
#       for y in range(9):
#             fig.add_subplot(9, 9, count)
#             plt.imshow(outputs[y,x])
#             plt.axis('off')
#             count = count +1











