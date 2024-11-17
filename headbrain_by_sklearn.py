import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import mean_squared_error
from sklearn.metrics  import r2_score
import csv
df = pd.read_csv('headbrain.csv')
print(df.head())

X = df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values

#modeL INITIALIZATION
reg = LinearRegression()

m = len(X)

X = X.reshape((m,1))

#model fitting
reg = reg.fit(X,Y)

#Y prediction 
Y_pred = reg.predict(X)

print(Y_pred)

#print(Y)

#model Evaluation
rmse = np.sqrt(mean_squared_error(Y,Y_pred))

r2 = r2_score(Y,Y_pred)

print(rmse)
print(r2)
