# A deep neural network model to detect fraud transactions in the
# credit card transaction data
# Data is unbalanced, oversample the fraud class

import time
from datetime import datetime
import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

if __name__ == "__main__":
    print ("\nStarting at :", str(datetime.now()))
    # Read data
    df = pd.read_csv('data/creditcard.csv')
    df.drop(['Time'], axis=1, inplace=True)
    # Density plot for the below columns for class =0/1 were similar
    cols_remove = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
    df.drop(cols_remove, axis=1, inplace=True)
    fraud = df[df['Class']==1]
    normal = df[df['Class']==0]

    # Take a fraction of normal data for training
    dfn = normal.sample(frac=0.15)
    f1 = fraud.sample(n=425)
    # Test set for the imbalanced class
    f2 = fraud.loc[set(fraud.index) - set(f1.index)]

    # Oversample fraud data
    f_over = pd.DataFrame()
    for _ in range(75):
        f_over = pd.concat([f_over, f1], ignore_index=True)
    print ("Shapes - Normal {}, Subsampled Fraud {}, Subsampled Nornal dfn {}".format(normal.shape, f1.shape, dfn.shape))

    # Training data
    df2 = pd.concat([dfn, f_over], ignore_index=True)
    print ("Shape of dfn and fraud concatenated: {}".format(df2.shape))
    df2 = shuffle(df2)
    df2 = df2.reset_index(drop=True)
    print ("df2['Class'].value_counts():", df2['Class'].value_counts())
    y2 = df2['Class'].values
    df2.drop(['Class'], axis=1, inplace=True)
    df2 = df2.values

    def baseline_model(n1=18, n2=18, lr=0.01):
        # Create model
        model = Sequential()
        # model.add(Dropout(0.2))
        model.add(Dense(n1, input_dim=18, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(n2, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(n2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        sgd = optimizers.SGD(lr=lr)
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    n1 = [36]
    n2 = [18]
    #lr = [0.1, 0.01, 0.001]
    param_grid = dict(n1=n1, n2=n2)
    #param_grid = dict()
    print ("param_grid defined")
    model = KerasClassifier(build_fn=baseline_model, nb_epoch=120, batch_size=5, verbose=2)
    print ("KerasClassifier defined ...")
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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

    print ("\n\nBest: %f using %s\n" % (results.best_score_, results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print ("%f (%f) with: %r" % (mean, stdev, param))
    
    print ("\n\nBest params:")
    pprint.pprint (results.best_estimator_.get_params())
    print ("\ncv results:")
    pprint.pprint (results.cv_results_)

