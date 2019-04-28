"""
Aaron Hunt
Homework 2 Problem 2

Predict label in hungarian dataset
Use processed.hungarian.data as argv[1]
Outputs accuracy
"""

import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

np.random.seed(2)

df = pd.read_csv(sys.argv[1], names=['age','sex','cp','trestbps','chol','fbs','restecg',
									 'thalach','exang','oldpeak','slope','ca','thal','num'])
df = df.astype(str)

# remove fields with a lot of missing data
df = df.drop(df[['slope','ca','thal']], axis=1)
df = df.drop(df[df.age == '?'].index)
df = df.drop(df[df.sex == '?'].index)
df = df.drop(df[df.cp == '?'].index)
df = df.drop(df[df.trestbps == '?'].index)
df = df.drop(df[df.chol == '?'].index)
df = df.drop(df[df.fbs == '?'].index)
df = df.drop(df[df.restecg == '?'].index)
df = df.drop(df[df.thalach == '?'].index)
df = df.drop(df[df.exang == '?'].index)
df = df.drop(df[df.oldpeak == '?'].index)
df = pd.concat([df, pd.get_dummies(df['sex'], prefix = 'sex')], axis = 1)
del df['sex']

df = df.astype(float)

X = df.values
# split training and testing
np.random.shuffle(X)
training, testing = X[:230,:], X[230:,:]

training_in = training[:,:]

training_labels = training[:,-3]


testing_in = testing[:,:]
testing_labels = testing[:,-3]

clf = MultinomialNB()
clf.fit(training_in, training_labels.ravel())
predicted_labels = clf.predict(testing_in)
acc = accuracy_score(testing_labels,predicted_labels)
print(acc)