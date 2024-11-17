import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
df = pd.read_csv('Iris.csv')
# print(df)

x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y = df['Species'].values

# print(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# print(x_train)
# print(y_train)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)


print(y_pred)
print(y_test)

cmf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cmf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score",metrics.f1_score(y_test,y_pred))