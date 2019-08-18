import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import model

def load_data(feature_file, label_file):
    features = pd.read_csv(feature_file)
    #Convert pandas DataFrame to numpy array
    features = features.values

    labels = pd.read_csv(label_file)
    #Convert labels from [m,] shape to [m,1] shape
    labels = np.reshape(labels.values, [-1, 1])

    return features, labels

train_features, train_labels = load_data("train_features.csv", "train_labels.csv")
test_features, test_labels = load_data("test_features.csv", "test_labels.csv")

# Number of features
n = train_features.shape[1]

optimizer, accuracy, X, Y, Y_hat = model.build_model(num_features=n)

with tf.Session() as sess: 
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
         
    # Iterating through all the epochs 
    for epoch in range(50001): 
        # Running the Optimizer 
        sess.run(optimizer, feed_dict = {X : train_features, Y : train_labels}) 
        
        if epoch % 1000 == 0:
            # Calculating cost on current Epoch 
            current_accuracy = sess.run(accuracy, feed_dict = {X : train_features, Y : train_labels}) 
            print("Accuracy:", current_accuracy * 100.0, "%")

    pred = sess.run(Y_hat, feed_dict = {X : test_features}) 
    test_accuracy = sess.run(accuracy, feed_dict = {X : test_features, Y : test_labels})
    print("Test accuracy:", test_accuracy * 100.0, "%")