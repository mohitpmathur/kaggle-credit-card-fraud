# Kaggle competition to detect fraud in credit
# card transactions

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.utils import shuffle


df = pd.read_csv('data/creditcard.csv')

Y = df['Class'].values
df.drop(['Time', 'Class'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.3, random_state=42)

params = {
	'learning_rate': [0.001]
}
gbc = GradientBoostingClassifier()
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = GridSearchCV(gbc, params, cv=3)
start = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - start
y_pred = clf.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print "Time Taken to fit model: ", time.strftime('%H:%M:%S', time.gmtime(elapsed))
print "AUC Score:", metrics.auc(fpr, tpr)
print "\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred)
print "\nClassification Report:", metrics.classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])

# THINGS TO TRY
# Resample the data to evenly distribute
# Divide 'Normal' data into 5 separate datasets, and use entire 'Fraud' data for each dataset
# and build model for each.
