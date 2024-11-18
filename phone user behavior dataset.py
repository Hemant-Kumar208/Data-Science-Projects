import pandas as pd
import sklearn.neighbors as KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('user_behavior_dataset.csv')
le = preprocessing.LabelEncoder()
df['Operating System'] = le.fit_transform(df['Operating System'])

x = df[['Operating System','Screen On Time (hours/day)','App Usage Time (min/day)','Number of Apps Installed','Data Usage (MB/day)','Age']].values
y = df[['User Behavior Class']].values
# print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# knn

# knn = KNeighborsClassifier.KNeighborsClassifier(n_neighbors=5) #assigns KNeighborsClassifier to variable knn
# knn.fit(x_train,y_train) #calls fit method on the knn variable

# y_pred = knn.predict(x_test)

# print(y_pred)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# cmf_matrix = metrics.confusion_matrix(y_test,y_pred)
# print(cmf_matrix)



# logestic 



logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cmf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cmf_matrix)