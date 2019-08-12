import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

def load_data():
    use_columns = [
        "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", 
        "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
        "Mode", "Mean", "Median", "Variance", "Tendency", "CLASS", "NSP" 
    ]
    source_data = pd.read_csv("CTG.csv", usecols=use_columns)
    #Remove any rows with missing values
    source_data = source_data.dropna()

    #The CLASS column is loaded as float64 by default.
    #We need to cast it as integer for it to work as a categorical column (from 1 to 10)
    source_data["CLASS"] = source_data["CLASS"].astype(int)

    # Split source into training (80%) and test data (20%)
    train_features = source_data.sample(frac=0.8, random_state=200)
    test_features = source_data.drop(train_features.index)

    # Extract the label column NSP
    train_labels = train_features.pop("NSP")
    test_labels = test_features.pop("NSP")

    #Do one hot encoding of NSP. Example: NSP 2 is [0, 1, 0]
    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)

    #Return pandas DataFrame and Series
    return train_features, train_labels, test_features, test_labels

def build_model(num_classes):
    numeric_features = [
        "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", 
        "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
        "Mode", "Mean", "Median", "Variance", "Tendency" 
    ]
    numeric_columns = [
        tf.feature_column.numeric_column(key=feature)
        for feature in numeric_features
    ]

    #The CLASS column values range from 1 to 10.
    class_categorical_column = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key="CLASS", 
                vocabulary_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            dimension=4)
    

    all_columns = numeric_columns + [class_categorical_column]

    print("All columns", all_columns)
    
    optimizer_adam = tf.train.AdamOptimizer(learning_rate=0.01)
    model = tf.estimator.DNNClassifier([9,9,3], 
        n_classes=num_classes,
        feature_columns=all_columns,  
        optimizer=optimizer_adam,
        model_dir="nn_classifier")
    
    return model

def train_model():
    train_features, train_labels, _, _ = load_data()
    #Number of classes
    num_classes = train_labels.shape[1]

    nn_classifier = build_model(num_classes)

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_features,
                                                            y=train_labels,
                                                            batch_size=65,
                                                            shuffle=True,
                                                            num_epochs=100)
    nn_classifier.train(input_fn = training_input_fn, steps=2000)

def test_model():
    _, _, test_features, test_labels = load_data()
    #Number of classes
    num_classes = test_labels.shape[1]

    nn_classifier = build_model(num_classes)

    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=test_labels,
                                                            batch_size=65,
                                                            shuffle=False,
                                                            num_epochs=1)
    result = nn_classifier.evaluate(input_fn = eval_input_fn, steps=2000)
    print(result)
    
train_model()