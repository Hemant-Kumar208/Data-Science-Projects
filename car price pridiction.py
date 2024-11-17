import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('Craigslist Car Dataset.csv')

# Handle missing values
df['odometer'] = df['odometer'].fillna(df['odometer'].median())
df['price'] = df['price'].fillna(df['price'].median())
df['condition'] = df['condition'].fillna(df['condition'].mode()[0])

# Convert 'condition' to numerical labels using LabelEncoder
label = preprocessing.LabelEncoder()
df['condition'] = label.fit_transform(df['condition'])

# Prepare the features and target variable
X = df[['odometer', 'condition', 'year']].values
Y = df['price'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

# Fit the model
Logreg = LinearRegression()
Logreg.fit(X_train, Y_train)

# Predictions and evaluation
Y_pred = Logreg.predict(X_test)
print(Y_pred)

# Compute R2 score
r2 = r2_score(Y_test, Y_pred)
print(f'R2 Score: {r2}')

