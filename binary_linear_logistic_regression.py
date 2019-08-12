import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

#Columns: Temp, Humidity
train_features = [
    # Hot and humid - rain predicted
    [86.0, 90.0],
    [94.0, 85.0],
    [91.0, 82.0],
    [84.0, 83.0],
    [90.0, 97.0],
    # Cold and dry - no rain
    [52.0, 67.0],
    [45.0, 58.0],
    [35.0, 62.0],
    [40.0, 61.0],
    [48.0, 72.0]
]

train_labels = [
    [1], # rain
    [1],
    [1],
    [1],
    [1],
    [0], # no rain
    [0],
    [0],
    [0],
    [0]
]

test_features = [
    # Hot and humid - rain predicted
    [83.0, 91.0],
    [92.5, 88.0],
    # Cold and dry - no rain
    [50.3, 63.0],
    [42.0, 59.0]
]

test_labels = [
    [1],
    [1],
    [0],
    [0]
]

# Number of features 2. Temp, Humidity
n = 2

# Input features
# There are n columns in the feature matrix  
X = tf.placeholder(tf.float32, [None, n]) 
  
# Real life prediction data
Y = tf.placeholder(tf.float32, [None, 1]) 
  
# Trainable Variable Weights 
W = tf.Variable(tf.zeros([n, 1])) 
  
# Trainable Variable Bias 
b = tf.Variable(tf.zeros([1])) 

# Hypothesis 
Y_hat =  tf.nn.sigmoid(tf.matmul(X, W) + b)
  
# Sigmoid Cross Entropy Cost Function 
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat))
  
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost) 

# Predicted value over 0.5 is rounded to 1 or True
correct_prediction = tf.equal(tf.round(Y_hat), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
 
with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
         
    # Iterating through all the epochs 
    for epoch in range(1001): 
        cost_per_epoch = 0
          
        # Running the Optimizer 
        sess.run(optimizer, feed_dict = {X : train_features, Y : train_labels}) 
        
        if epoch % 100 == 0:
            # Calculating cost on current Epoch 
            current_cost = sess.run(cost, feed_dict = {X : train_features, Y : train_labels}) 
            current_accuracy = sess.run(accuracy, feed_dict = {X : train_features, Y : train_labels}) 
            print("Cost:", current_cost, "Accuracy:", current_accuracy * 100.0, "%")
    
    pred = sess.run(Y_hat, feed_dict = {X : test_features}) 
    test_accuracy = sess.run(accuracy, feed_dict = {X : test_features, Y : test_labels})
    print("Test accuracy:", test_accuracy * 100.0, "%")