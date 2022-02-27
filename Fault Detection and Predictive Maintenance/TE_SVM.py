# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:56:59 2019

@author: Kaushal
"""

#importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training dataset
dataset_train = pd.read_csv('20train.csv')
X_train = dataset_train.iloc[:,:-1].values
y_train = dataset_train.iloc[:,52].values

#importing the testing dataset
dataset_test = pd.read_csv('20test.csv')
X_test = dataset_test.iloc[:,:-1].values
y_test = dataset_test.iloc[:,52].values

#Encoding class label for multi-class classification
from sklearn.preprocessing import LabelEncoder
labelencoder_y_train = LabelEncoder()
labelencoder_y_test = LabelEncoder()
y_train = labelencoder_y_train.fit_transform(y_train)
y_test = labelencoder_y_test.fit_transform(y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(C = 100, kernel = 'poly',degree = 4, gamma = 0.09)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test, y_pred)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#improving model performance (tuning of hyperparameters)
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.009, 0.01, 0.02, 0.03, 0.04, 0.05,  0.06, 0.07, 0.08]},
              {'C': [1, 10, 100], 'kernel': ['poly'], 'degree': [3, 4], 'gamma': [0.009, 0.01, 0.02, 0.03, 0.04, 0.05,  0.06, 0.07, 0.08]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

