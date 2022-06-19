"""
NUMBERS RECOGNITION TRAINING
"""
import os
import glob
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
from tensorflow.keras import layers,models
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def trainModelMnist():
      '''
      Create a model to recognize mnist numbers from kaggle https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist/notebook
      Pretrain model with mnist data and data augmentation

      Returns:
            model
      '''
      (x_train, y_train) , (x_val, y_val) = keras.datasets.mnist.load_data()
      x_train = x_train / 255
      x_val = x_val / 255
      
      x_train = x_train.reshape(-1,28,28,1)
      x_val = x_val.reshape(-1,28,28,1)
      
      model = models.Sequential([
            layers.Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)),
            layers.BatchNormalization(),
            layers.Conv2D(32,kernel_size=3,activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Conv2D(64,kernel_size=3,activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64,kernel_size=3,activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(10, activation='softmax')      
            ])
      
      model.summary()

      model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      
      datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.10,
            height_shift_range=0.1,
            width_shift_range=0.1)
      
      history = model.fit(datagen.flow(x_train, y_train), validation_data=(x_val, y_val), epochs=3)
      
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('mnist model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()
      
      return model


def trainModelRealDate(model):
      '''
      Train model from numbers extract from sudoku image

      Arguments:
            model: pretrain model from mnist
      Returns:
            model and his history
      '''
      x_train, y_train = buildRealData()
      x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
      
      #lock the first layers and train only the 2 last dense layers
      model.layers[0].trainable = False
      model.layers[1].trainable = False
      model.layers[2].trainable = False
      model.layers[3].trainable = False
      model.layers[4].trainable = False
      model.layers[5].trainable = False
      
      model.layers[7].trainable = False
      model.layers[8].trainable = False
      model.layers[9].trainable = False
      model.layers[10].trainable = False
      model.layers[11].trainable = False
      model.layers[12].trainable = False
      datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.10,
            height_shift_range=0.1,
            width_shift_range=0.25) #an hight width shift because our data vary a lot in this direction -> drastically improve performance 
      history = model.fit(datagen.flow(x_train, y_train), validation_data=(x_val,y_val), epochs=50)
      
      return model, history

def buildRealData():
      '''
      Retreive data from numbers extract from sudoku image

      Returns:
            x_train: input 
            y_train: output 
      '''
      x_train = []
      y_train = []
      for i in range(0,10):
            files = glob.glob('../sudoku_img/'+str(i)+'/*.jpg')
            for file in files:            
                  x_train.append(np.array(Image.open(file).convert('L')))                                                                          
                  y_train.append(i) 
                    
      x_train = np.array(x_train)
      y_train = np.array(y_train)
      
      x_train = x_train / 255
      x_train = abs(x_train - 1)
      x_train = x_train.reshape(-1,28,28,1)
           
      return x_train, y_train
	       

def trainNumberRecognition():
      '''
      Create model and train it with mnist and our own data

      Returns model and his history
      '''
      model = trainModelMnist()
      model, history = trainModelRealDate(model)
      return model, history


model, history = trainNumberRecognition()

#save model and weight
os.makedirs("output", exist_ok=True)
model.save("output/model_numbersReading")
model.save_weights("output/model_numbersReading_weights")

#plot loss train vs validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




#test data augmentation parameter

# file = glob.glob('sudoku_img/3/462.jpg')
# image = np.array(Image.open(file[0]).convert('L'))

# image = image / 255
# image = abs(image - 1)
# image = image.reshape(-1,28,28,1)

# # create image data augmentation generator
# datagen = ImageDataGenerator(zoom_range = 0.20)
# # prepare iterator
# it = datagen.flow(image, batch_size=1)
# # generate samples and plot
# for i in range(9):
#  	# define subplot
#  	plt.subplot(330 + 1 + i)
#  	# generate batch of images
#  	batch = it.next()
#  	# convert to unsigned integers for viewing
#  	image2 = batch[0].reshape(28,28)
#  	# plot raw pixel data
#  	plt.imshow(image2)
# # show the figure
# plt.show()

