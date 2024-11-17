import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
df = pd.read_csv('diabetes.csv')
# print(df)
x = df[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
y = df['Outcome'].values
print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# print(x_train)
# print(y_train)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print(y_pred)
# print(y_test)

cmf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cmf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score",metrics.f1_score(y_test,y_pred))
