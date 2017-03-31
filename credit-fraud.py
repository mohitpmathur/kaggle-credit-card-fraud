# Kaggle competition to detect fraud in credit
# card transactions

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('data/creditcard.csv')

Y = df['Class'].values
df.drop(['Time', Class], axis=1, inplace=True)
X = df.values

def baseline_model():
	# Create model
	model = Sequential()
	model.add(Dense(29, input_dim=29, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['auc'])
	return model

X_train, y_train, X_test, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)