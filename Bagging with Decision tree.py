#Bagging with decision tree
#import libraries
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#syntetic data
x, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=20,
)
print(x, y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)
#Initialise the decision tree classifier
base_estimator = DecisionTreeClassifier(random_state=20)

#Initialise the bagging classifier
bagging_model = BaggingClassifier(
    estimator = base_estimator,
    n_estimators=100,          #no. of trees
    random_state=20
)

#train the bagging classifier
bagging_model.fit(x_train,y_train)

#make prediction
y_pred = bagging_model.predict(x_test)
#Evaluate model
accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy: {accuracy}")
