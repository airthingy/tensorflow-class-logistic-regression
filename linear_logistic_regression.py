import pandas as pd
import tensorflow.compat.v1 as tf

def load_data(feature_file, label_file):
    features = pd.read_csv(feature_file)
    #Convert pandas DataFrame to numpy array
    features = features.values

    labels = pd.read_csv(label_file)
    #Do one hot encoding: Convert 0 to [1, 0] and 1 to [0, 1]
    labels = pd.get_dummies(labels["Survived"]).values

    return features, labels, features.shape[1]

train_features, train_labels, n = load_data("train_features.csv", "train_labels.csv")
test_features, test_labels, n = load_data("test_features.csv", "test_labels.csv")

#Number of classes (survived, died)
K = 2

# There are n columns in the feature matrix  
X = tf.placeholder(tf.float32, [None, n]) 
  
# Since this is a binary classification problem, 
# each Y will be like [0, 1] or [1, 0]
Y = tf.placeholder(tf.float32, [None, 2]) 
  
# Trainable Variable Weights 
W = tf.Variable(tf.zeros([n, K])) 
  
# Trainable Variable Bias 
b = tf.Variable(tf.zeros([K])) 

# Hypothesis 
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b)) 
  
# Sigmoid Cross Entropy Cost Function 
cost = tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = Y_hat, labels = Y) 
  
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost) 

with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
      
    # Lists for storing the changing Cost and Accuracy in every Epoch 
    cost_history, accuracy_history = [], [] 
      
    # Iterating through all the epochs 
    for epoch in range(10001): 
        cost_per_epoch = 0
          
        # Running the Optimizer 
        sess.run(optimizer, feed_dict = {X : train_features, Y : train_labels}) 
        
        if epoch % 100 == 0:
            
            # Calculating accuracy on current Epoch 
            correct_prediction = tf.equal(tf.argmax(Y_hat, 1), 
                                            tf.argmax(Y, 1)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
                                                    tf.float32)) 
            # Calculating cost on current Epoch 
            current_cost = sess.run(cost, feed_dict = {X : train_features, Y : train_labels}) 
            print(accuracy.eval({X : test_features, Y : test_labels}) * 100)

      
    Weight = sess.run(W) # Optimized Weight 
    Bias = sess.run(b)   # Optimized Bias 