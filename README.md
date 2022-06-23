# Solve sudoku from it image
This is a project to practice what I had learn about CNN.
The goal is to solve sudoku from it image only by using machine learning/deep learning model.

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

The outline isn't perfect but all the numbers are fine inside.

AP metrics ...

<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuGridExtract.png">
<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/numbersExtract.png">
<br clear="left"/>
Then we extract the grid by warping the image to the box size and split each cell into unique image

## Step 2: Read numbers
## Step 3: Solve sudoku grid
