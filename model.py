import numpy as np
import tensorflow.compat.v1 as tf

def build_model(num_features):
    #Number of classes. In a binary classification
    #there's only one class.
    K = 1

    # Feature matrix  
    X = tf.placeholder(tf.float32, [None, num_features]) 
    
    # Since this is a binary classification problem, 
    # each Y will be mx1 dimension
    Y = tf.placeholder(tf.float32, [None, K]) 
    
    # Trainable Variable Weights 
    W = tf.Variable(tf.zeros([num_features, K])) 
    
    # Trainable Variable Bias 
    b = tf.Variable(tf.zeros([K])) 

    # Hypothesis 
    logits = tf.matmul(X, W) + b
    Y_hat =  tf.nn.sigmoid(logits)
    
    # Sigmoid Cross Entropy Cost Function 
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
                            labels=Y, logits=logits))   
    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost) 

    #Round a prediction over 0.5 to 1 and less to 0. Then compare
    #with actual outcome.
    correct_prediction = tf.equal(tf.round(Y_hat), Y) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    
    return optimizer, accuracy, X, Y, Y_hat