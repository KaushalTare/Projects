# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:02:27 2019

@author: Kaushal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Training_data = pd.read_csv ('20train.csv')
Testing_data = pd.read_csv ('20test.csv')

X_train , y_train = Training_data.iloc[:, :-3].values,Training_data.iloc[:,52].values
X_test , y_test = Testing_data.iloc[:, :-3].values,Testing_data.iloc[:,52].values


y_enc_train = pd.get_dummies(y_train)
y_enc_test = pd.get_dummies(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 26, init = 'uniform', activation = 'relu', input_dim = 52))

# Adding the output layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train_scale, y_enc_train, batch_size = 100, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test_scale)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_enc_test.values.argmax(axis=1), y_pred.argmax(axis=1))

accuracy = accuracy_score(y_enc_test.values.argmax(axis=1), y_pred.argmax(axis=1))

