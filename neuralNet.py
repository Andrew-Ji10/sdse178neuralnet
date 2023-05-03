import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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
scaler = StandardScaler()
scaler.fit(Xtrain.astype('float64'))
#Xtrain2 = scaler.transform(Xtrain.astype('float64'))
#Xtest2 = scaler.transform(Xtest)

mlClassify2 = MLPRegressor(solver="sgd", random_state=rngSeed, max_iter=100000) #lbfgs serves to converge faster than the default adam technique
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

print(ytest.to_numpy)
#plt.plot(yhatml2)
#plt.plot(ytest)
#plt.plot(yhatml)
#plt.show()