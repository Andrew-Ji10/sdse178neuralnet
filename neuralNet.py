import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rngSeed = 45332

file = pd.read_csv(r"C:\Users\jiand\SDSE178\FinalProject\sdse178neuralnet\carData.csv")

length = np.array([1, 2, 3, 4])

## Clean the data & split
desiredcolumns = file.columns[5:19]
practicalFile = file.get(desiredcolumns)

print(desiredcolumns)

Xtrain, Xtest, ytrain, ytest = train_test_split(practicalFile[desiredcolumns[1:]],
                                                practicalFile[desiredcolumns[0]], 
                                                test_size=0.2, random_state=rngSeed)
"""
print(Xtrain.astype('float64').dtypes)
print(ytrain.dtypes)
mlClassify = MLPRegressor(solver="lbfgs", random_state=rngSeed, max_iter=10000) #lbfgs serves to converge faster than the default adam technique

model = Pipeline([
    ('scaler' , StandardScaler()) ,
    ('model' , mlClassify)
])

model.fit(Xtrain.astype('float64'), ytrain)

yhatml = model.predict(Xtest)
mlAcc = mean_squared_error(ytest, yhatml)
mlError = mean_absolute_error(ytest, yhatml)
print(ytest, yhatml)
print(mlAcc, mlError)

## test the affects of sdg based solving scaling

mlClassify2 = MLPRegressor(solver="sgd", random_state=rngSeed, max_iter=100000) #sdg serves to converge faster than the default adam technique
model2 = Pipeline([
    ('scaler' , StandardScaler()) ,
    ('model' , mlClassify2)
])

model2.fit(Xtrain, ytrain)

yhatml2 = model2.predict(Xtest)
mlAcc2 = mean_squared_error(ytest, yhatml2)
mlError2 = mean_absolute_error(ytest, yhatml2)
print(ytest, yhatml2)
print(mlAcc2, mlError2)

## test the affects of Adam based solving scaling

mlClassify3 = MLPRegressor(solver="adam", random_state=rngSeed, max_iter=100000) #adam serves to converge faster than the default adam technique
model3 = Pipeline([
    ('scaler' , StandardScaler()) ,
    ('model' , mlClassify3)
])

model3.fit(Xtrain, ytrain)

yhatml3 = model3.predict(Xtest)
mlAcc3 = mean_squared_error(ytest, yhatml3)
mlError3 = mean_absolute_error(ytest, yhatml3)
print(ytest, yhatml3)
print(mlAcc3, mlError3)
"""
#Using gridsearchCV to improve the accuracy of the best case, sdg algorythm

from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()
scaler.fit(Xtrain)

scaledXtrain = scaler.transform(Xtrain)
scaledXtest = scaler.transform(Xtest)


cvfolds = 3

mlRegress4 = MLPRegressor(max_iter=100000, random_state=rngSeed)

"""#feature selection

SFS = SequentialFeatureSelector(mlRegress4, scoring='neg_mean_squared_error')
print("here")
SFS.fit(scaledXtrain, ytrain)
print(SFS.get_support(), "done")

#gridsearch"""

param_grid = {"solver": ["lbfgs", "sgd", "adam"], "learning_rate" : ['constant', 'invscaling', 'adaptive']}

gs = GridSearchCV(mlRegress4, param_grid, scoring='neg_mean_absolute_error', cv=cvfolds, refit='neg_mean_absolute_error')
gs = gs.fit(scaledXtrain, ytrain)

yhatml2v2 = gs.predict(scaledXtest)
mlAcc2v2 = mean_squared_error(ytest, yhatml2v2)
mlError2v2 = mean_absolute_error(ytest, yhatml2v2)

print(mlAcc2v2, mlError2v2, gs.get_params())

plt.plot(ytest.to_numpy(), 'b')
plt.plot(yhatml2v2, "r--" )
plt.show()
