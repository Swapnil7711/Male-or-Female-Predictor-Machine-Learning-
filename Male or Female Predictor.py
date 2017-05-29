# Import required packages

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
import math

# data required for training model
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Choose the training classifier

TreeClassifier = tree.DecisionTreeClassifier()
SVMClassifier  = svm.SVC()
GaussianNbClassifier = GaussianNB()
NeighborClassifier = neighbors.KNeighborsClassifier()


# Train the classifier

TreeClassifier = TreeClassifier.fit(X,Y)
SVMClassifier  = SVMClassifier.fit(X,Y)
GaussianNbClassifier = GaussianNbClassifier.fit(X,Y)
NeighborClassifier  = NeighborClassifier.fit(X,Y)

# Testing data. Always use the seperate datasets for trainig and testing
# to avoid overfitting

X_Test = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40]]
Y_Test = ['male', 'male', 'female', 'female', 'male']

#Test our trained Classifier

TreeClassifier_prediction = TreeClassifier.predict(X_Test)
SVMClassifier_prediction  = SVMClassifier.predict(X_Test)
GaussianNbClassifier_prediction = GaussianNbClassifier.predict(X_Test)
NeighborClassifier_prediction  = NeighborClassifier.predict(X_Test)

# print the accuracy score for each model

print("The accuracy score for Tree Classifier is : ",accuracy_score(Y_Test,TreeClassifier_prediction)*100)
print("The accuracy score for SVM Classifier is : ",accuracy_score(Y_Test,SVMClassifier_prediction)*100);
print("The accuracy score for Gaussian Classifier is : ",accuracy_score(Y_Test,GaussianNbClassifier_prediction)*100);
print("The accuracy score for Neighbor Classifier is : ",accuracy_score(Y_Test,NeighborClassifier_prediction)*100);


# define ScoreArray dictionary for storing model and it's accuracy

ScoreArray = {'TreeClassifier':accuracy_score(Y_Test,TreeClassifier_prediction)*100,'SVMClassifier':accuracy_score(Y_Test,SVMClassifier_prediction)*100,
              'GaussianNbClassifier':accuracy_score(Y_Test,GaussianNbClassifier_prediction)*100,'NeighborClassifier':accuracy_score(Y_Test,NeighborClassifier_prediction)*100}


# Print the most effective model with it's accuracy score 
print('most effective algorithm is :',max(ScoreArray, key=ScoreArray.get) ,'with score :',ScoreArray[max(ScoreArray, key=ScoreArray.get)])







