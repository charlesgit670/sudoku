"""
SUDOKU SOLVER PREDICTION
"""
import csv
import os

from tensorflow.keras import models
import numpy as np



def modelSolver():
      '''
      Load model and weight

      Returns model
      '''
      model = models.load_model(os.path.abspath(os.path.dirname( __file__ ))+"/../model_training/output/model_sudokuSolver")
      model.load_weights(os.path.abspath(os.path.dirname( __file__ ))+"/../model_training/output/model_sudokuSolver_weights")
      return model

def sudoku_solver(grid):
      '''
      Solve sudoku grid
      Slightly improve result with deleting bad prediction and predict again
      
      Arguments:
            grid -- sudoku grid unsolved with blanck fill with 0, array (9,9) 
            model -- model already trained
      Returns:
            grid: sudoku grid solved, array (9,9)
      '''
      model =  modelSolver()
      count = 0
      while 0 in grid and count < 3:  
            grid = sudokuPrediction(grid,model)
            grid = deleteBadPredictionInGrid(grid)
            count += 1
      return grid
      

def sudokuPrediction(grid, model):
      '''
      Fill sudoku with prediction from our model

      Arguments:
            grid -- sudoku grid unsolved with blanck fill with 0, array (9,9) 
            model -- model already trained
      Returns:
            grid: sudoku grid solved, array (9,9)
      '''
      mask = np.zeros([9,9,9])
      for i in range(9):
            for j in range(9):
                  if grid[i,j] != 0:
                        mask[i,j,:] = 0
                  else:
                        mask[i,j,:] = 1
            
      grid = grid.reshape(1,9,9,1)
      while 0 in grid[0,:,:,0]:
            
            output = model.predict(grid)
            output = output.squeeze()
            output = output*mask
           
            cellMaxScore = output.max(axis=2)
            cellMaxArg = np.argmax(output, axis=2)+1
            
            coord = np.argwhere(cellMaxScore==cellMaxScore.max())[0].reshape(2,)
      
            grid[0,coord[0],coord[1],0] = cellMaxArg[coord[0],coord[1]]
            mask[coord[0],coord[1],:] = 0
      
      return grid.reshape(9,9)     

def checkSudokuValid(grid, test=False):
      '''
      Check if sudoku is well resolved

      Arguments:
            grid: sudoku grid, array (9,9)
            test: True if we want to return false when there is an error detected, False if we want to throw an assert error
      Returns:
            True if correct, else throw an assert error or False if test=True
      '''
      isCorrectNumber = False
      for x in grid:
            error = False
            for y in x:                  
                  if y in [1,2,3,4,5,6,7,8,9]:
                        isCorrectNumber = True
                  else:
                        isCorrectNumber = False
                        error = True
                        break
            if error:
                  break
      if test:
            if isCorrectNumber == False:
                  print(grid)
                  return False
      else:
            assert isCorrectNumber == True, 'there is number not in list 1,2,3,4,5,6,7,8,9'
      
      
      for i in range(9):
            setRow = set()
            setColumn = set()
            for j in range(9):                              
                  setRow.add(grid[i,j])
                  setColumn.add(grid[j,i])
            if test:
                  if len(setRow) != 9 | len(setColumn) != 9:
                        print(grid)
                        return False
            else:
                  assert len(setRow) == 9, ('there is non unique value in row ',i)
                  assert len(setColumn) == 9, ('there is non unique value in column ',i)
      
      for n in range(3):
            for p in range(3):
                  setBox = set()
                  for i in range(3):
                        for j in range(3):                  
                              setBox.add(grid[n*3+i,p*3+j])
                  if test:
                        if len(setBox) != 9:
                              print(grid)
                              return False
                  else:
                        assert len(setBox) == 9, ('there is non unique value in box ',[n,p])
      return True

def deleteBadPredictionInGrid(grid):
      '''
      delete bad values in sudoku grid
      
      Returns:
            grid: clean from bad values if exist
      '''
      for i in range(9):
            row = []
            column = []
            for j in range(9):  
                  if grid[i,j] in row:
                        grid[i,row.index(grid[i,j])] = 0
                        grid[i,j] = 0                      
                  else:
                        row.append(grid[i,j])
                  if grid[j,i] in column:                        
                        grid[column.index(grid[j,i]),i] = 0
                        grid[j,i] = 0     
                  else:
                        column.append(grid[j,i])
                                   
      for n in range(3):
            for p in range(3):
                  box = []
                  for i in range(3):
                        for j in range(3):
                              if grid[n*3+i,p*3+j] in box:
                                    grid[n*3+box.index(grid[n*3+i,p*3+j])//3,p*3+box.index(grid[n*3+i,p*3+j])%3] = 0
                                    grid[n*3+i,p*3+j] = 0
                              else:
                                   box.append(grid[n*3+i,p*3+j]) 
                                                                              
      return grid
      
      
      
def testAccuracy():
      '''
      Estimate accuracy on a data set with easy level of sudoku

      '''
      grids = []
      model_sudokuSolver = modelSolver()
      with open('../sudokuGrid/sudoku.csv') as sudoku_data:
            rows = csv.reader(sudoku_data, delimiter=',')
            for count, row in enumerate(rows):
                  if count == 0:
                        continue
                  if count == 100:
                        break
                  grids.append(np.array(list(map(int, list(row[0])))).reshape(9,9))
      grids = np.array(grids).reshape(-1,9,9,1)
      
      totalNumberOfGrid = grids.shape[0]
      count = 0
      for grid in grids:   
            output = sudoku_solver(grid, model_sudokuSolver)
            if checkSudokuValid(output, True): 
                  count += 1
      
      print('Accuracy over ',totalNumberOfGrid,' example : ',count/totalNumberOfGrid)
      
def testAccuracy2():
      '''
      Estimate accuracy on a data set with different level of sudoku
      
      '''
      grids = []
      model_sudokuSolver = modelSolver()
      with open('../sudokuGrid/sudoku-3m.csv') as sudoku_data:
            rows = csv.reader(sudoku_data, delimiter=',')
            for count, row in enumerate(rows):
                  if count == 0:
                        continue
                  if count == 100:
                        break
                  grids.append(np.array(list(map(int, list(row[1].replace('.','0'))))).reshape(9,9))
      grids = np.array(grids).reshape(-1,9,9,1)
      
      totalNumberOfGrid = grids.shape[0]
      count = 0
      for grid in grids:   
            output = sudoku_solver(grid, model_sudokuSolver)
            if checkSudokuValid(output, True): 
                  count += 1
      
      print('Accuracy over ',totalNumberOfGrid,' example : ',count/totalNumberOfGrid)
      
      
# testAccuracy2()







      


  