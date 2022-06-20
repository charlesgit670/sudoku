"""
SUDOKU SOLVER TRAINING
Read Data -> create model -> train model -> save model and weight -> plot loss train/validation
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models,Model
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split

from model_prediction.sudoku_solver_prediction import sudoku_solver, checkSudokuValid

x_train = []
y_train = []

#extract and transform data
with open('../sudokuGrid/sudoku-3m.csv') as sudoku_data:
      rows = csv.reader(sudoku_data, delimiter=',')
      nbrRow = 3000000 #nbr row to extract
      for count, row in enumerate(rows):
            if count == 0:
                  continue
            if count == nbrRow:
                  break
            x_train.append(np.array(list(map(int, list(row[1].replace('.','0'))))).reshape(9,9))
            y_train.append(np.array(list(map(int, list(row[2])))).reshape(9,9))
            if count % (nbrRow//20) == 0:
                  print('completion : ',(count*100)//nbrRow,'%')
     
                 
x_train = np.array(x_train).reshape(-1,9,9,1)
y_train = np.array(y_train).reshape(-1,9,9)

y_train = y_train-1 # [1-9] -> [0-8], class start from 0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train) #split train/validation set (0.8/0.2)
                 
def model():
      '''
      custom u-net architecture -> bad result

      Returns model
      '''
      inputs = Input((9,9,1))
      conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
      conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
      conv1 = layers.BatchNormalization()(conv1)
      skip_connection = conv1
      conv1 = layers.MaxPooling2D((3, 3), strides=3)(conv1)
      layers.Dropout(0.5),
      
      conv2 = layers.Conv2D(filters=192, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
      conv2 = layers.Conv2D(filters=192, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
      conv2 = layers.BatchNormalization()(conv2)
      
      conv3 = layers.Conv2DTranspose(filters=192, kernel_size=(3, 3),strides=3, padding='same')(conv2)
      conv3 = layers.concatenate([conv3, skip_connection],axis=3)
      
      conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
      conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
      conv3 = layers.BatchNormalization()(conv3)
      
      conv4 = layers.Conv2D(filters = 9, kernel_size=(1, 1))(conv3)
      conv4 = layers.Activation('softmax')(conv4) 
      
      model = Model(inputs=inputs, outputs=conv4)
      
      return model


def model2():
    '''
    model retreive from https://github.com/shivaverma/Sudoku-Solver/blob/master/model.py -> has the best result
    
    Returns model
    '''  

    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(81*9))
    model.add(layers.Reshape((9,9,9)))
    model.add(layers.Activation('softmax'))
    
    return model

def model3():
      '''
      try to add rule of sudoku game in the model with row,colum and box shape filter -> bad result
    
      Returns model
      '''  
      inputs = Input((9,9,1))
      
      convRow     = layers.Conv2D(filters=64, kernel_size=(9, 1), activation='relu')(inputs)     
      convRow     = layers.BatchNormalization()(convRow)
      
      convColumn  = layers.Conv2D(filters=64, kernel_size=(1, 9), activation='relu')(inputs)
      convColumn  = layers.BatchNormalization()(convColumn)
      
      convBox     = layers.Conv2D(filters=64, kernel_size=(3, 3), strides = 3, activation='relu')(inputs)
      convBox     = layers.BatchNormalization()(convBox)
      
      convRow     = layers.Conv2DTranspose(filters=64, kernel_size=(9, 1), activation='relu')(convRow)
      convColumn  = layers.Conv2DTranspose(filters=64, kernel_size=(1, 9), activation='relu')(convColumn)
      convBox     = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides = 3, activation='relu')(convBox)
      
      concat = layers.concatenate([convRow, convColumn, convBox])

      conv = layers.Conv2D(filters = 9, kernel_size=(1, 1))(concat)
      conv = layers.Activation('softmax')(conv) 
      
      model = Model(inputs=inputs, outputs=conv)
      
      return model

modelSolver = model2()  #load model
modelSolver.summary()   #display architecture of the model

#we use sparse_categorical_crossentropy because one hot encoding doesn't improve accuracy and we run outofmemory with large dataset (1 million sudoku grid)
modelSolver.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
history = modelSolver.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=64, epochs=3)

#calculate accuracy
# testGrid = x_val[1:100]    
# totalNumberOfGrid = testGrid.shape[0]
# count = 0
# for grid in testGrid:   
#       output = sudoku_solver(grid)
#       if checkSudokuValid(output, True): 
#             count += 1      
# print('Accuracy over ',totalNumberOfGrid,' example : ',count/totalNumberOfGrid)

#save model and weight
os.makedirs("output", exist_ok=True)
modelSolver.save("output/model_sudokuSolver")
modelSolver.save_weights("output/model_sudokuSolver_weights")

#plot loss train vs validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

    
