"""
Aaron Hunt
Homework 2, Problem 1

Predict class of nuclear cortex data using KNN
Use Data_Cortex_Nuclear.xls as argv[1]
"""

import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(452) # seed for reproducibility

# import the data as a dataframe
df = pd.read_excel(sys.argv[1])

# remove columns with a lot of missing data
df = df.drop(df[['BAD_N','BCL2_N','pCFOS_N','H3AcK18_N','EGR1_N','H3MeK4_N']], axis = 1)
del df['MouseID']

# hot encode Genotype, Treatment, Behavior and class
df = pd.concat([df, pd.get_dummies(df['Genotype'], prefix = 'Genotype')], axis = 1)
del df['Genotype']
df = pd.concat([df, pd.get_dummies(df['Treatment'], prefix = 'Treatment')], axis = 1)
del df['Treatment']
df = pd.concat([df, pd.get_dummies(df['Behavior'], prefix = 'Behavior')], axis = 1)
del df['Behavior']
df = pd.concat([df, pd.get_dummies(df['class'], prefix = 'class')], axis = 1)
del df['class']

# remove rows with NaN values
df = df.dropna()

# convert dataframe to np array
X = df.values
# split training and testing
np.random.shuffle(X)
training, testing = X[:150,:], X[150:,:]

training_in = training[:,:]
training_labels = training[:,-8:]

testing_in = testing[:,:]
testing_labels = testing[:,-8:]

# build the model
knn = KNeighborsClassifier()
knn.fit(training_in,training_labels)

# test the model
predicted_labels = knn.predict(testing_in)

# accuracy of model
acc = accuracy_score(testing_labels, predicted_labels)
print(acc)