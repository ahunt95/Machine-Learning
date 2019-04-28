'''
Aaron Hunt
Problem 2

Use bezdekIris.data as argv[1]
'''

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

np.random.seed(23452345)

# Make data frame and hot encode classes
df = pd.read_csv(sys.argv[1], names=['s_length','s_width','p_length','p_width','class'])
df = pd.concat([df, pd.get_dummies(df['class'], prefix = 'class')], axis = 1)
del df['class']
# Split training and testing data
training = df.sample(frac = 9/10, axis = 0)
testing = df.drop(training.index)

# training input
training_attributes = training.values
# training output
training_class = training[['class_Iris-setosa','class_Iris-versicolor','class_Iris-virginica']].values
# testing input
testing_attributes = testing.values
# testing output
testing_class = testing[['class_Iris-setosa','class_Iris-versicolor','class_Iris-virginica']].values

# Build the neural net
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='linear'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_attributes, training_class, epochs=150, batch_size=10, verbose = 0)


# Test accuracy
scores = model.evaluate(testing_attributes, testing_class, verbose = 0)

print(scores[1])
