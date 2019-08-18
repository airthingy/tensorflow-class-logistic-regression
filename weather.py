import tensorflow.compat.v1 as tf
import model

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

optimizer, accuracy, X, Y, Y_hat = model.build_model(
    num_features=2)

with tf.Session() as sess: 
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
         
    # Iterating through all the epochs 
    for epoch in range(1001): 
        # Running the Optimizer 
        sess.run(optimizer, 
            feed_dict = {X : train_features, Y : train_labels}) 
        
        if epoch % 100 == 0:
            # Calculating cost on current Epoch 
            current_accuracy = sess.run(accuracy, 
                feed_dict = {X : train_features, Y : train_labels}) 
            print("Accuracy:", current_accuracy * 100.0, "%")
    
    prediction = sess.run(Y_hat, 
        feed_dict = {X : test_features}) 
    test_accuracy = sess.run(accuracy, 
        feed_dict = {X : test_features, Y : test_labels})
    print("Test accuracy:", test_accuracy * 100.0, "%")
    for pair in zip(prediction, test_labels):
        print("Predicted:", pair[0], "Actual:", pair[1])