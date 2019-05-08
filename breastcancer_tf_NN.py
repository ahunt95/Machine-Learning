
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

np.random.seed(123)

data = pd.read_csv('breast-cancer-wisconsin.data',header=None)
data_array = data.values
#remove rows with missing vals
good_rows = np.invert(np.any(data_array=='?',1))
data = data_array[good_rows]
x= data[:,1:10].astype(np.float32)
y=data[:,10].astype(int)
le = LabelEncoder()
le.fit(y)
y=le.transform(y)
y=np_utils.to_categorical(y)
(x_train,x_test,y_train,y_test) = train_test_split(x,y,test_size=.2)

hidden_nodes=5
num_y_train = y_train.shape[1]
batch_size = 100
num_features = x_train.shape[1]
learning_rate = .01

graph = tf.Graph()
with graph.as_default():
    #Data
    tf_train_dataset = tf.placeholder(tf.float32,shape=[None,num_features])
    tf_train_labels = tf.placeholder(tf.float32,shape=[None,num_y_train])
    tf_test_dataset = tf.constant(x_test)

    #weights and biases
    layer1_weights = tf.Variable(tf.truncated_normal([num_features,hidden_nodes]))
    layer1_biases = tf.Variable(tf.zeros([hidden_nodes]))
    layer2_weights = tf.Variable(tf.truncated_normal([hidden_nodes,num_y_train]))
    layer2_biases = tf.Variable(tf.zeros([num_y_train]))
    #Three-layer netowrk
    def three_layer_network(data):
        input_layer = tf.matmul(data,layer1_weights)
        hidden = tf.nn.relu(input_layer+layer1_biases)
        output_layer = tf.matmul(hidden,layer2_weights)+layer2_biases
        return output_layer

    #Model Scores
    model_scores = three_layer_network(tf_train_dataset)

    #Loss

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =tf_train_labels,logits=model_scores ))
    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #is learning rate the proportionality term in backpropagation?
    
    #Predictions 
    train_prediction =tf.nn.softmax(model_scores)
    test_prediction = tf.nn.softmax(three_layer_network(tf_test_dataset))

# train
def accuracy(predictions,labels):
    preds_correct_boolean = np.argmax(predictions,1) == np.argmax(labels,1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = correct_predictions/predictions.shape[0]
    return accuracy

num_steps = 10001

with tf.Session(graph=graph) as sess:
    #tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())
    for step in range(num_steps):
        offset = (step*batch_size) % (y_train.shape[0]- batch_size)
        minibatch_data = x_train[offset:(offset+batch_size),:]
        minibatch_labels = y_train[offset:(offset+batch_size)] #feel like should be column : too
        feed_dict = {tf_train_dataset:minibatch_data,tf_train_labels:minibatch_labels}
        _,lo,predictions = sess.run([optimizer,loss,train_prediction],feed_dict = feed_dict)

        if step % 1000 == 0:
            print("Minibatch loss at step {0}: {1}".format(step,lo))
    print(accuracy(test_prediction.eval(),y_test))