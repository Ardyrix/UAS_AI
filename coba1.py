# -*- coding: utf-8 -*-
"""
Created on Sun Jun 9 18:20:17 2019

@author: BOT
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Fungsi Import Dataset
def importdata():
    haberman_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/haberman/haberman.data',
    sep= ',', header = None)

# Print Format Dataset
    print ("Dataset Lenght: ", len(haberman_data))
    print ("Dataset Shape: ", haberman_data.shape)

# Print Dataset
    print ("Dataset: ",haberman_data.head())
    return haberman_data

# Fungsi Split Dataset
def splitdataset(haberman_data):

# Memisahkan Variabel Target
    X = haberman_data.values[:, 0:2]
    Y = haberman_data.values[:, 3]

# Split Dataset Train dan Test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.6, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

# Menjalankan Train dengan Entropy
def train_using_entropy(X_train, X_test, y_train):

# Decision Tree dengan Entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 10)

# Menjalankan Train
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Fungsi Membuat Prediksi
def prediction(X_test, clf_object):

# Prediksi menggunakan giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Fungsi Menghitung Akurasi
def cal_accuracy(y_test, y_pred):
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

def main():

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
     
    print("Results Using Entropy:")
# Prediksi Menggunakan Entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
     
     
# Calling main function
if __name__=="__main__":
    main()