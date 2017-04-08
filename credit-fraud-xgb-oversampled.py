# Quick things to try for the Kaggle credit-card fraud detection
# Data is unbalanced, undersample the normal class

import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

if __name__ == '__main__':
    print "\nStarting at :", str(datetime.now())
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
    print "Shapes - Normal {}, Oversampled Fraud {}, Subsampled Nornal dfn {}".format(normal.shape, f_over.shape, dfn.shape)
    
    # Training data
    df2 = pd.concat([dfn, f_over], ignore_index=True)
    print "Shape of dfn and fraud concatenated: {}".format(df2.shape)
    df2 = shuffle(df2)
    df2 = df2.reset_index(drop=True)
    print "df2['Class'].value_counts() - ", df2['Class'].value_counts()
    y2 = df2['Class']
    df2.drop(['Class'], axis=1, inplace=True)

    #learning_rate = [0.0055, 0.01]
    #learning_rate = [0.01]
    #n_estimators = [1250, 1500]
    #n_estimators = [1250]
    #max_depth = range(3, 12, 2)
    min_child_weight= range(1,6,2)
    param_grid = dict(
        # learning_rate=learning_rate,
        # n_estimators=n_estimators,
        #max_depth=[7, 9],
        #min_child_weight=[1, 3],
        # gamma = [i/10.0 for i in range(0,5)]
        gamma=[0]
        )
    
    xgb = XGBClassifier(
        learning_rate=0.01, 
        n_estimators=1250, 
        max_depth=9, 
        min_child_weight=3, 
        silent=True,
        seed=7)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    clf = GridSearchCV(
        xgb,
        param_grid, 
        cv=kfold, 
        scoring='roc_auc', 
        verbose=2, 
        n_jobs=-1)

    start = time.time()
    grid_result = clf.fit(df2, y2)
    end = time.time()
    elapsed = end - start
    print "Time taken to fit the model:", time.strftime("%H:%M:%S", time.gmtime(elapsed))

    print "\nBest: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print "Mean %f, std (%f) with: %r" % (mean, stdev, param)

    # Test data
    dfn2 = normal.sample(frac=0.01)
    #f2 = fraud.sample(frac=0.5)
    print "Shapes - Subsampled Normal dfn2 {}, Normal {}, Subsampled fraud {}\n".format(dfn2.shape, normal.shape, f2.shape)
    dft2 = pd.concat([dfn2, f2], ignore_index=True)
    dft2 = shuffle(dft2)
    dft2 = dft2.reset_index(drop=True)
    print "dft2['Class'].value_counts() -\n", dft2['Class'].value_counts()
    yt2 = dft2['Class']
    dft2.drop(['Class'], axis=1, inplace=True)

    # predict classification
    y_pred = clf.predict(dft2)

    # Display metrics
    fpr, tpr, _ = metrics.roc_curve(yt2, y_pred)
    print "AUC Score:", metrics.auc(fpr, tpr)
    
    print "\nConfusion Matrix:\n", metrics.confusion_matrix(yt2, y_pred)
    print "\nClassification Report:", metrics.classification_report(yt2, y_pred, target_names=['Normal', 'Fraud'])
    print "\nBest Estimator:\n", clf.best_estimator_
    print "\nBest Parameters:\n", clf.best_params_

    # plot results
    # scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    # for i, value in enumerate(learning_rate):
    #     pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    # pyplot.legend()
    # pyplot.xlabel('n_estimators')
    # pyplot.ylabel('Log Loss')
    # pyplot.savefig('n_estimators_vs_learning_rate.png')

    """
    OUTPUT:

        AUC Score: 0.954872139478

        Confusion Matrix:
        [[2841    2]
        [   6   61]]

        Classification Report:              precision    recall  f1-score   support

            Normal       1.00      1.00      1.00      2843
            Fraud       0.97      0.91      0.94        67

        avg / total       1.00      1.00      1.00      2910


        Best Estimator:
        XGBClassifier(base_score=0.5, colsample_bytree=1, gamma=0, learning_rate=0.01,
            max_delta_step=0, max_depth=9, min_child_weight=3, missing=None,
            n_estimators=1250, nthread=-1, objective='binary:logistic', seed=7,
            silent=True, subsample=1)

        Best Parameters:
        {'gamma': 0}

    """
