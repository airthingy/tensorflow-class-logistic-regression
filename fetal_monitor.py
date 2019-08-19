import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

def build_model(num_features, num_classes):
    # Number of features
    n = num_features

    #Number of classes.
    K = num_classes

    # There are n columns in the feature matrix  
    X = tf.placeholder(tf.float32, [None, n]) 
    
    # Label is mxK matrix.
    Y = tf.placeholder(tf.float32, [None, K]) 
    
    # Weights. nxK matrix 
    W = tf.Variable(tf.truncated_normal([n, K], stddev=0.001)) 
    
    # Bias. One for each class. 
    b = tf.Variable(tf.truncated_normal([K], stddev=0.001)) 

    # Hypothesis uses softmax.
    #It is a mxK matrix
    logits = tf.matmul(X, W) + b
    Y_hat =  tf.nn.softmax(logits)

    # Cost function for softmax hypotheses function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                            labels=Y, logits=logits))  
    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost) 

    #Index of the highest predicted value is taken as 
    #the final decision.
    correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    return optimizer, X, Y, Y_hat, cost, accuracy

def load_data():
    use_columns = [
        "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", 
        "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
        "Mode", "Mean", "Median", "Variance", "Tendency", "CLASS", "NSP" 
    ]
    source_data = pd.read_csv("CTG.csv", usecols=use_columns)
    #Remove any rows with missing values
    source_data = source_data.dropna()
    #Encode categorical feature CLASS using indicator variables.
    source_data = pd.get_dummies(source_data, columns=["CLASS"], drop_first=True)

    # Split source into training (80%) and test data (20%)
    train_features = source_data.sample(frac=0.8, random_state=200)
    test_features = source_data.drop(train_features.index)

    # Extract the label column NSP
    train_labels = train_features.pop("NSP")
    test_labels = test_features.pop("NSP")

    #Do one hot encoding of NSP. Example: NSP 2 is [0, 1, 0]
    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)

    #Return numpy array formatted data
    return train_features.values, train_labels.values, test_features.values, test_labels.values

train_features, train_labels, test_features, test_labels = load_data()
# Number of features
num_features = train_features.shape[1]
#Number of classes. Should be 3.
num_classes = train_labels.shape[1]

optimizer, X, Y, Y_hat, cost, accuracy = build_model(num_features, num_classes)

with tf.Session() as sess: 
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
    
    # Iterating through all the epochs 
    for epoch in range(100001): 
        # Running the Optimizer 
        sess.run(optimizer, feed_dict = {X : train_features, Y : train_labels}) 
    
        # Calculating cost on current Epoch 
        if epoch % 1000 == 0:
            current_cost = sess.run(cost, feed_dict = {X : train_features, Y : train_labels}) 
            current_accuracy = sess.run(accuracy, feed_dict = {X : train_features, Y : train_labels}) 
            print("Cost:", current_cost, "Accuracy:", current_accuracy * 100.0, "%")
            if current_accuracy > 0.9:
                break
    
    test_accuracy = sess.run(accuracy, feed_dict = {X : test_features, Y : test_labels})
    print("Test accuracy:", test_accuracy * 100.0, "%")
    prediction = sess.run(Y_hat, feed_dict = {X : test_features})
    #Convert prediction and labels to NSP values
    prediction = np.argmax(prediction, axis=1) + 1
    expected = np.argmax(test_labels, axis=1) + 1
    for pair in zip(prediction, expected):
        print("Predicted:", pair[0], "Expected:", pair[1])