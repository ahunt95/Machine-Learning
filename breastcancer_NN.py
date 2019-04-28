"""
Aaron Hunt
Problem 1

Run with breast_cancer_wisconsin.data as argv[1]
"""

import os 
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull,'w') 
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

sys.stderr = stderr

np.random.seed(42) # Seed

# Make data frame
df = pd.read_csv(sys.argv[1], names = ['id','ct','ucsi','ucs','ma','secs','bn','bc',
									   'nn','m','class'])
df = df.drop('bn',1)
df = df.drop('id',1)

# Hot-encode the output for orthogonality
df = pd.concat([df, pd.get_dummies(df['class'], prefix = 'class')], axis = 1)
del df['class']

# Split training and testing
training = df.sample(frac = 9/10, axis = 0)
testing = df.drop(training.index)

training_out = training[['class_2', 'class_4']].values
training_in = training.values
testing_out = testing[['class_2', 'class_4']].values
testing_in = testing.values

# Build the neural net
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_in, training_out, epochs=150, batch_size=10, verbose = 0)

# Test accuracy
scores = model.evaluate(testing_in, testing_out, verbose = 0)
print(scores[1])