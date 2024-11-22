import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
df = pd.read_csv('Customer Purchasing Behaviors.csv') #headbrain.csv is a extension

X = df['annual_income'].values
Y = df['purchase_amount'].values

mean_X = np.mean(X)
mean_Y = np.mean(Y)

numer = 0
denom = 0
for i in range(len(X)):
    numer += (X[i] - mean_X) * (Y[i] - mean_Y)
    denom += (X[i] - mean_X) ** 2

m = numer / denom
c = mean_Y - (m * mean_X)

max_X = np.max(X)+5
min_X = np.min(X)-5
a = np.linspace(min_X, max_X)
b = m * a + c

plt.scatter(X, Y, color="yellow")
plt.plot(a, b, color="red")
plt.xlabel("annual_income")
plt.ylabel("purchase_amount")
plt.title("annual_income vs purchase_amount Regression Line")
plt.show()

x = 25000
y = m*x+c
print(y)
print(df)