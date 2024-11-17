import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
df = pd.read_csv('Social_Network_Ads.csv')
le = preprocessing.LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
# print(df)
x = df[['Gender', 'Age', 'EstimatedSalary']].values
y = df['Purchased'].values
# print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# print(x_train)
# print(y_train)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print(y_pred)
print(y_test)

cmf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cmf_matrix)

import seaborn as sns

class_names =[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#creating heatmap
sns.heatmap(pd.DataFrame(cmf_matrix), annot=True, cmap="Greens",fmt='d',annot_kws={"size":10})
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.4)
plt.xlabel("Actual label")
plt.ylabel('Predicted value')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score",metrics.f1_score(y_test,y_pred))
