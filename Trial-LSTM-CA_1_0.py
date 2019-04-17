# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:02:35 2019

@author: Paule Carelle
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from numpy import array
from numpy import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import concatenate
from sklearn.metrics import confusion_matrix

# split a multivariate sequence into samples
def split_sequences(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
#Data selection
train = pd.read_csv('CA_crew1_0.csv', index_col='time')
train.drop(['Unnamed: 0'], axis=1, inplace=True)
train.index.name = 'time'

#Sample of the data
Sample = train.sample(frac=0.2,  replace=True, random_state=0)
Sample.sort_index(inplace=True)
Sample.drop(['crew','experiment'], axis=1, inplace=True)

# Encoding categorical data
values = Sample.values
labelencoder = LabelEncoder()
values[:, 24] = labelencoder.fit_transform(values[:, 24])
values = values.astype('float32') 

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
values[:, 0:24] = scaler.fit_transform(values[:, 0:24])

# Drop columns we don't want to predict
Final= pd.DataFrame(values)

#Set inputs and outputs
inputs = Final.iloc[:, 0:24].values
output = Final.iloc[:, 24].values
inputs = inputs.reshape((len(inputs), 24))
output = output.reshape((len(output), 1))

# horizontally stack columns
dataset = hstack((inputs, output))

# Number of time steps- equivalant to 35 seconds
n_steps = 44*35
n_features = 25
# convert into input/output
X, y = split_sequences(dataset, n_steps)

# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Initialising the model
classifier = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
verbose, epochs, batch_size = 0, 50, 150
n_steps, n_features, n_outputs = n_steps, X_train.shape[2], y_train.shape[0]
classifier.add(LSTM(units = 100, return_sequences = True, 
                    input_shape = (n_steps, n_features)))
classifier.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(n_outputs, activation='softmax'))

# Compiling the model
model= classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

# Fitting the model to the Training set
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
               validation_data=(X_test, y_test))

# plot model
pyplot.plot(model.history['loss'], label='train')
pyplot.plot(model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Final evaluation of the model
accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
print("Accuracy: %.2f%%" % (accuracy[1]*100))

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = accuracy(X_train, X_test, y_train, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# III- Making the predictions 

# Getting prediction for test set
y_pred = model.predict(X_test)
print(y_pred)

# invert scaling for forecast
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
inv_ypred = concatenate((y_pred, X_test[:, 1:]), axis=1)
inv_ypred = scaler.inverse_transform(inv_ypred)
inv_ypred = inv_ypred[:,0]

# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Plot the prediction
pyplot.plot(inv_ypred)
pyplot.plot(inv_y)
pyplot.show()

####### NEEDS A LITTLE BIT OF WORK BUT YOU CAN RUN IT AND SEE IF IT WORKS #########
#IV-Prediction on real Test Data

#load test data
test = pd.read_csv('test.csv', index_col='time')
test.drop(['Unnamed: 0'], axis=1, inplace=True)
test.index.name = 'time'

test_id = test['id']
test.drop(['id', 'crew', 'experiment'], axis=1, inplace=True)


# Feature Scaling
values_test = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
values_test[:,0:2] = scaler.fit_transform(values[:,0:24])

#Predict probabilities of Ids in Test data
Test= pd.DataFrame(values)
pred = model.predict_proba(Test)
sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])
sub['id'] = test_id
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
sub.to_csv("Test_prob.csv", index=False)
