In this repo we solve various classification (logistic regression) problems. We use both linear regression and neural network.

Create a folder called **workshop/logistic-regression** anywhere in your hard drive. All code will go here.

# Workshop - Basic Binary Linear Logistic Regression
In this very simple workshop we will predict if it will rain given the temperature and humidity. We purposely keep the problem simple. The goal is to focus on how to create a model using Tensorflow for linear logistic regression. Specifically, we will pay attention to:

- The dimension of feature matrix
- The dimension of label and prediction matrix
- How to define the hypotheses (output) function
- How to model cost
- How to calculate accuracy of prediction

## Create the Model
In **workshop/logistic-regression** create file called ``model.py``.

You should be familiar with the mathematics of sigmoid function and how to compute cost for it. We will use low level Tensorflow API to build the model. 

Add this code.

```python
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
    b = tf.Variable(0.0) 

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
```

Save your file.

# Create the Weather Data Set
To keep things simple we will hard code tempreature, humidity and chance of rain. In real life, of course, this will be loaded from file and there will be many more features.

In **workshop/logistic-regression** create a file called ``weather.py``.

Add this code (freel free to copy and paste from below).

```python
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
```

>In binary classification the label matrix always has ``mx1`` dimension. Where ``m`` is the number of samples. During training the labels have values of either 0 or 1. During prediction the values range from 0 to 1. We can round the values to 0 and 1 to get a yes or no answer.

## Create the Model
Add this code.

```python
optimizer, accuracy, X, Y, Y_hat = 
    model.build_model(num_features=2)
```

## Do Training and Prediction

Add this code. This is very similar to regression except ``prediction`` is not a real number. It is a ``mx1`` matrix of real numbers ranging from 0 to 1.

```python
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
```

Save your file.

## Run Code
Run the code.

```
python3 weather.py
```

You should get near 100% accuracy.

# Workshop - Titanic Survivability Prediction (Optional)
This is a binary logistic regression problem. The model will be same as what we have already developed for rain prediction. Code will be practically same as for the weather problem, except of course we will load real life data from file. This is why the workshop is optional. There is, however, one key difference. Several features, such as class of travel and gender, are categorical in nature. In this lab we will learn to create dummy (or indicator) features for them.

## Instructor Demo
Instructor should use Jupyter notebook to demo how ``pd.get_dummies()`` works.

## Prepare Data
Source data was downloaded from:

```
https://www.kaggle.com/c/titanic
```

Open ``train.csv`` and observe the columns we will use.

- **Survived** - This is the prediction. Value is 0 (died) or 1 (survived)
- **Pclass** - Class of travel. This is a categorical column. Values are 1, 2 and 3 representing First, Second and Third class travel. Just because they have numerical values don't be tempted to use them as a numerical feature. This could have been represented as "First", "Second" and "Third". 
- **Sex** - Gender. Values are "male" and "female". This is a categorical feature.
- **Age**.
- **SibSep** - Number of siblings/spouses aboard the Titanic
- **Parch** - Number of parents/children aboard the Titanic
- **Fare** - Fare paid for ticket

The remaining columns like passenger name and ticket number are deemed irrelevant.

Copy ``prepare_data.py`` from the solution folder. Open it and observe how ``pd.get_dummies()`` is used to create indicator features.

Run the file.

```
python3 prepare_data.py
```

This will create the following files.

- train_features.csv - Open this file and observe how the indictor columns Pclass_2, Pclass_3, etc. are created.
- train_labels.csv
- test_labels.csv
- test_features.csv

## Load Data
In **workshop/logistic-regression** create a file called ``titanic.py``.

Add this code.

```python
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

train_features, train_labels = load_data(
    "train_features.csv", "train_labels.csv")
test_features, test_labels = load_data(
    "test_features.csv", "test_labels.csv")

```

## Create the Model
Add this code.

```python
# Number of features
n = train_features.shape[1]

optimizer, accuracy, X, Y, Y_hat = model.build_model(num_features=n)
```

## Train and Predict
Add this code. This is practically same as the rain prediction problem. Except we run training for a larger number of epochs.

```python
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
```

Save changes.

Run the code.

```python
python3 titanic.py
```

Test should produce over 90% accuracy.

# Multi-class Linear Logistic Regression
In this problem we need to classify a case in one of several possible categories. For example, we need to evaluate the risk of lending a sum of money to someone as Low, Medium and High.

## Fetal Monitoring Complication Prediction
Cardiotocography is used to monitor fetal heartbeat and  uterine contractions during pregnancy. Various metrics are used to predict complications like hypoxia and acidosis.

Prediction falls under three categories:
- Normal: No hypoxia/acidosis
- Suspect: Low probability of hypoxia/acidosis
- Pathologic: High probability of hypoxia/acidosis, requires immediate action

## About the Data
Data is available as an Excel file from:

```
https://archive.ics.uci.edu/ml/datasets/cardiotocography
```

Data was extracted from the ``Raw Data`` worksheet of that Excel file to the ``CTG.csv`` file.

Description of the columns:

- LB - FHR baseline (beats per minute) 
- AC - # of accelerations per second 
- FM - # of fetal movements per second 
- UC - # of uterine contractions per second 
- DL - # of light decelerations per second 
- DS - # of severe decelerations per second 
- DP - # of prolongued decelerations per second 
- ASTV - percentage of time with abnormal short term variability 
- MSTV - mean value of short term variability 
- ALTV - percentage of time with abnormal long term variability 
- MLTV - mean value of long term variability 
- Width - width of FHR histogram 
- Min - minimum of FHR histogram 
- Max - Maximum of FHR histogram 
- Nmax - # of histogram peaks 
- Nzeros - # of histogram zeros 
- Mode - histogram mode 
- Mean - histogram mean 
- Median - histogram median 
- Variance - histogram variance 
- Tendency - histogram tendency 
- CLASS - FHR pattern class code (1 to 10). This is a categorical feature.
- NSP - fetal state class code (1=normal; 2=suspect; 3=pathologic). This is what we need to predict.

## Build the Model
In **workshop/logistic-regression** folder create a file called ``fetal_monitor.py``. Add this code.

```python
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
```

Take a moment to take stock of the dimensions of the following tensors:
- X - Features 
- Y - Labels 
- Y_hat - Predictions 
- W - Weights 
- b - Biases

## Load Data
Add this function.

```python
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
```

Note the following aspects of the code:
- How we create indicator variables for the ``CLASS`` feature.
- How we one hot encode the labels

## Run Training and Test
Add this code.

```python
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
```

Most of this should be straightforward. But pay attention to how we are getting the predicted NSP values.

Save changes and run the code.

```
python3 fetal_monitor.py
```

You should get about 90% accuracy.