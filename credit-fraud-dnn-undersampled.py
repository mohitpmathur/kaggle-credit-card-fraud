# Quick things to try for the Kaggle credit-card fraud detection
# Data is unbalanced, undersample the normal class

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Read data
df = pd.read_csv('data/creditcard.csv')
fraud = df[df['Class']==1]
normal = df[df['Class']==0]

# Take a fraction of normal data for training
dfn = normal.sample(frac=0.15)
f1 = fraud.sample(n=425)
# Test set for the imbalanced class
f2 = fraud.loc[set(fraud.index) - set(f1.index)]
print ("Shapes - Normal {}, Subsampled Fraud {}, Subsampled Nornal dfn {}".format(normal.shape, f1.shape, dfn.shape))

# Training data
df2 = pd.concat([dfn, f1], ignore_index=True)
print ("Shape of dfn and fraud concatenated: {}".format(df2.shape))
df2 = shuffle(df2)
df2 = df2.reset_index(drop=True)
print ("df2['Class'].value_counts():", df2['Class'].value_counts())
y2 = df2['Class'].values
df2.drop(['Class'], axis=1, inplace=True)
df2 = df2.values

def baseline_model(n1=30, n2=30):
	# Create model
	model = Sequential()
	model.add(Dense(n1, input_dim=30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(n2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

n1 = [30, 45]
n2 = [30, 15]
param_grid = dict(n1=n1, n2=n2)
#param_grid = dict()
print ("param_grid defined")
model = KerasClassifier(build_fn=baseline_model, nb_epoch=120, batch_size=5, verbose=2)
print ("KerasClassifier defined ...")
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
print ("StratifiedKFold defined ...")
clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='recall', cv=kfold)
print ("GridSearchCV defined ...")
start = time.time()
results = clf.fit(df2, y2)
end = time.time()
elapsed = end - start
print ("Time taken to fit the model:", time.strftime("%H:%M:%S", time.gmtime(elapsed)))

# Test data
dfn2 = normal.sample(frac=0.01)
#f2 = fraud.sample(frac=0.5)
print ("Shapes - Subsampled Normal dfn2 {}, Normal {}, Subsampled fraud {}".format(dfn2.shape, normal.shape, f2.shape))
dft2 = pd.concat([dfn2, f2], ignore_index=True)
dft2 = shuffle(dft2)
dft2 = dft2.reset_index(drop=True)
print ("dft2['Class'].value_counts() -", dft2['Class'].value_counts())
yt2 = dft2['Class']
dft2.drop(['Class'], axis=1, inplace=True)
dft2 = dft2.values

# predict classification
y_pred = clf.predict(dft2)

# Display metrics
fpr, tpr, _ = metrics.roc_curve(yt2, y_pred)
print ("AUC Score:", metrics.auc(fpr, tpr))
print ("\nConfusion Matrix:\n", metrics.confusion_matrix(yt2, y_pred))
print ("\nClassification Report:", metrics.classification_report(yt2, y_pred, target_names=['Normal', 'Fraud']))
