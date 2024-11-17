import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('fruit_chat.xlsx')
# print(df)


le= preprocessing.LabelEncoder()
df['Size'] = le.fit_transform(df['Size'])


x = df[['Weight (grams)','Color Intensity (0-10)','Size']].values
y = df['Fruit Type'].values

# print(x,y)
# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# print(x_train)
# print(y_train)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
new = [[140,7,0]]
y_pred = knn.predict(new)

print(y_pred)
print(new)
