import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

def load_data(feature_file, label_file):
    features = pd.read_csv(feature_file)
    #Convert pandas DataFrame to numpy array
    features = features.values

    labels = pd.read_csv(label_file)
    #Shape each label as [1] or [0]
    #labels will now be mx1 where m is number of samples
    labels = np.reshape(labels.values, [-1, 1])

    return features, labels

train_features, train_labels = load_data("train_features.csv", "train_labels.csv")
test_features, test_labels = load_data("test_features.csv", "test_labels.csv")

# Number of features
n = train_features.shape[1]

#Number of classes. In a binary classification
#there's only one class. survived=1 died=0
K = 1

# There are n columns in the feature matrix  
X = tf.placeholder(tf.float32, [None, n]) 
  
# Since this is a binary classification problem, 
# each Y will be like [0, 1] or [1, 0]
Y = tf.placeholder(tf.float32, [None, K]) 
  
# Trainable Variable Weights 
W = tf.Variable(tf.zeros([n, K])) 
  
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
 
with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
         
    # Iterating through all the epochs 
    for epoch in range(50001): 
        cost_per_epoch = 0
          
        # Running the Optimizer 
        sess.run(optimizer, feed_dict = {X : train_features, Y : train_labels}) 
        
        if epoch % 1000 == 0:
            # Calculating cost on current Epoch 
            current_cost = sess.run(cost, feed_dict = {X : train_features, Y : train_labels}) 
            current_accuracy = sess.run(accuracy, feed_dict = {X : train_features, Y : train_labels}) 
            print("Cost:", current_cost, "Accuracy:", current_accuracy * 100.0, "%")

    
    pred = sess.run(Y_hat, feed_dict = {X : test_features}) 
    test_accuracy = sess.run(accuracy, feed_dict = {X : test_features, Y : test_labels})
    print("Test accuracy:", test_accuracy * 100.0, "%")