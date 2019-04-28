from keras.models import Sequential
from keras.layers import Dense
import tensorflow
import pandas as pd
import numpy as np

np.random.seed(42)

# Import data
df = pd.read_csv('abalone.data')

# Make dummies for Sex variable
df = pd.concat([df, pd.get_dummies(df['Sex'], prefix = 'Sex')], axis = 1)
del df['Sex']

# Split training and testing
training = df.sample(frac = 9/10, axis = 0)
testing = df.drop(training.index)

training_out = training['Rings'].values
training_in = training.values

testing_out = testing['Rings'].values
testing_in = testing.values

# Build the model
model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Fit
model.fit(training_in, training_out, epochs=150, batch_size=10)

# Test accuracy and print score
scores = model.evaluate(testing_in, testing_out)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))