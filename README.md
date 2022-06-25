# Solve sudoku from his image
This is a project to practice what I had learn about CNN.
The goal is to solve sudoku from his image only by using machine learning/deep learning model.

The project is split into 3 differents step:
- Detection of the sudoku grid -> extract and warp grid image -> split each cell of the grid
- Read numbers from images
- Solve sudoku
## Step 1: Extract sudoku grid and numbers
To properly separate each numbers from the sudoku grid. I need to determine as precisely as possible the sudoku's outline. To achieve that, semantic segmentation seem to be the best choice here (of course there is non ML technique that can achieve it too). After some research, I decide to use the library detectron2 from Facebook AI Research. This library is easy to use and have multiple model for object detection, semantic or keypoint detection. In addition, each model has an already pretrained weight that can be loaded before running our data.
I used mask_rcnn_R_50_FPN_3x for segmentation because it's the lighter model with 3.0 Go used and it can run on my GPU [models documentation](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).

<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudoku.jpg">
<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuSegmentation.png">
<br clear="left"/>

Left the input image and right the output after segmentation. It worked very well !
<br/>
The outline isn't perfect but all the numbers are fine inside and the grid is detected with a very hight confidence (100% here). To represent how well the model work, we usually use AP metric.

<img align="left" width="680" height="90" src="https://github.com/charlesgit670/sudoku/blob/main/result/APmetric.JPG?raw=true">
<br clear="left"/>

AP (average precision) is based on precision/recall metric. Precision means how accurate are our prediction and Recall represent the prediction that are missing.
If we inscrease the confidence threshold, the prediction become more accurate => Precision increase but the Recall decrease because we miss more prediction.
To obtain AP we average over different confidence threshold.<br/>
For object detection or segmentation, we use in addition IoU for Intersection over Union. It represent how well the boundary boxex predicted compared to labeled boxes with a value bewtween 0 to 1. If we choose a IoU of 0.5, the prediction is considered correct if IoU >= 0.5 otherwise is incorrect. AP50 is for a IoU of 0.5, AP75 for 0.75 etc.
AP is equivalent at mAP (mean Average Precision) in detectron2 with an average over different value of IoU between 0.5 and 0.95 with a step of 0.05. For more detail you can consult this [blog](https://blog.roboflow.com/mean-average-precision/), or this [video](https://www.youtube.com/watch?v=FppOzcDvaDI).<br/>
Here we got a perfect score of 100 ! This problem is too easy for our model because the grid detected is very big. This score means that we miss 0 prediction on 40 samples with an hight confidence and IoU.

<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuGridExtract.png">
<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/numbersExtract.png">
<br clear="left"/>
Then we choose the segmentation with the largest area because we consider only one grid per image. We warp the image to the box size predicted and split each cell into unique image.

## Step 2: Read numbers
## Step 3: Solve sudoku grid
