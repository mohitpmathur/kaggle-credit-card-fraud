# Kaggle competition to detect fraud in credit
# card transactions

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('data/creditcard.csv')

Y = df['Class'].values
df.drop(['Time', 'Class'], axis=1, inplace=True)
X = df.values

def baseline_model():
	# Create model
	model = Sequential()
	model.add(Dense(29, input_dim=29, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=120, batch_size=5, verbose=2)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print "AUC Score:", metrics.auc(fpr, tpr)
print "\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred)
print "\nClassification Report:", metrics.classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])

# THINGS TO TRY
# Resample the data to evenly distribute
# Divide 'Normal' data into 5 separate datasets, and use entire 'Fraud' data for each dataset
# and build model for each.
