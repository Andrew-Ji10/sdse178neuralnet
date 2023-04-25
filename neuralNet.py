import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rngSeed = 45332

file = pd.read_csv(r"C:\Users\jiand\SDSE178\FinalProject\sdse178neuralnet\carData.csv")

length = np.array([1, 2, 3, 4])

## Clean the data & split
desiredcolumns = file.columns[5:19]
practicalFile = file.get(desiredcolumns)

Xtrain, Xtest, ytrain, ytest = train_test_split(practicalFile[desiredcolumns[1:]],
                                                practicalFile[desiredcolumns[0]], 
                                                test_size=0.2, random_state=rngSeed)

print(Xtrain)