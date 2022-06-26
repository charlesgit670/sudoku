# Solve sudoku from his image
This is a project to practice what I had learn about CNN.
The goal is to solve sudoku from his image only by using machine learning/deep learning model.

The project is split into 3 differents step:
- Detection of the sudoku grid -> extract and warp grid image -> split each cell of the grid
- Read digits from images
- Solve sudoku
## Step 1: Extract sudoku grid and digits
To properly separate each digits from the sudoku grid. We need to determine as precisely as possible the sudoku's outline. To achieve that, semantic segmentation seem to be the best choice here (of course there is non ML technique that can achieve it too). After some research, I decided to use the library detectron2 from Facebook AI Research. This library is easy to use and have multiple model for object detection, semantic or keypoint detection. In addition, each model has an already pretrained weight that can be loaded before running our data. I retreived from internet 200 images split into 160 train and 40 test that I labeled by hand with [labelme](https://datagen.tech/guides/image-annotation/labelme/).
I used mask_rcnn_R_50_FPN_3x for segmentation because it's the lighter model with 3.0 Go used and it can run on my GPU [models documentation](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).

<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudoku.jpg">
<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuSegmentation.png">
<br clear="left"/>

Left the input image and right the output after segmentation. It worked very well !
<br/>
The outline isn't perfect but all the digits are fine inside and the grid is detected with a very hight confidence (100% here). To represent how well the model work, we usually use AP metric.

<img align="left" width="680" height="90" src="https://github.com/charlesgit670/sudoku/blob/main/result/APmetric.JPG">
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

## Step 2: Read digits
Now, to read digits that we extracted before, we need another CNN model. I started with LeNet5 architecture with the famous mnist dataset (handwritten digit) for pretrain our model before training it on 800 digits extracted from the sudoku images with our previous model. Unfortunately the results were not perfect because our digits are not well centered and the border of the grid on some images disturb the prediction. To improve result, I found a model with better performance on [kaggle](https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist/notebook) that use data augmentation to make the model more robust. For the training part on our images, we increased the horizontal shift on data augmentation because there are more variance due to a non perfect segmentation of the grid.

<img align="left" width="333" height="250" src="https://github.com/charlesgit670/sudoku/blob/main/result/LossNumbersRecognition.png">
<img align="left" width="231" height="220" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuUnsolved.JPG">
<br clear="left"/>

The results are very good with an accuracy of 98% and no overfitting as shown in the figure on the left (there is no gap bewtween train and val loss). Nevertheless the model still struggle with blurry images. A future improvement can be made by adding blurry to the data augmentation.

## Step 3: Solve sudoku grid
Finally, it's time to solve the sudoku. I try a few models following my intuition with an output like Unet architecture, 9x9 (grid) x9 filter for each digits probabilities. The data used are retreive from [kaggle](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) which contains 3 millions sudoku with their complete solutions and different levels of difficuty. Even if we predict digit one by one with the highter probability instead of solving it in one time, we fully solved it with an accuracy close to 0% (yes, it's pretty bad). After that, I found a simple model with better results on [github](https://github.com/shivaverma/Sudoku-Solver/blob/master/model.py).

<img align="left" width="390" height="260" src="https://github.com/charlesgit670/sudoku/blob/main/result/LossSudokuSolver.png">
<br clear="left"/>

We check that we are not overfitting our model (no gap between train and val loss)

<img align="left" width="351" height="15" src="https://github.com/charlesgit670/sudoku/blob/main/result/AccuracyOver99MultipleDifficultiesExample.JPG">
<img align="left" width="345" height="14" src="https://github.com/charlesgit670/sudoku/blob/main/result/AccuracyOver99EasyExample.JPG">
<br clear="left"/>

On the left, the accuracy for a 99 samples from the 3 millions data with different level of difficulty and on the right the accuracy of another [data set](https://www.kaggle.com/datasets/bryanpark/sudoku) with easy level of difficulty only. The results are good on easy level only.

<img align="left" width="150" height="140" src="https://github.com/charlesgit670/sudoku/blob/main/result/suodkuSolved.JPG">
<br clear="left"/>

After applying the model on our previous extracted digits, we succesfully solved our sudoku. But what happen when it failed to solve it ?

<img align="left" width="150" height="140" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuUnsolved2.JPG">
<img align="left" width="150" height="140" src="https://github.com/charlesgit670/sudoku/blob/main/result/sudokuSolved2.JPG">
<br clear="left"/>

Here an example of failure. We delete each wrong value and replace them by 0. Even if we try to predict the latest numbers, the result is still the same.<br/>
This is an astonishing result ! It's very easy to finish the sudoku but the model can't. We can conclude that the model didn't understand the logic of the sudoku and that it probably solved it using brute force with our huge dataset of 3 millions grid of sudoku.

## Conclusion
We saw that CNN can be very efficient in detecting pattern like digits and didn't need a lot of data if it is well pretrained. In addition, data augmentation can significantly increase the robustness of the model. But CNN seems unable to understand logic in data like the sudoku rules.
