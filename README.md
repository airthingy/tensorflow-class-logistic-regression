In this repo we solve various classification (logistic regression) problems. We use both linear regression and neural network.

Create a folder called **workshop/logistic-regression**. All code will go here.

# Workshop - Basic Binary Linear Logistic Regression
In this very simple workshop we will predict if it will rain given the temperature and humidity. We purposely keep the problem simple. The goal is to focus on how to create a model using Tensorflow for linear logistic regression. Specifically, we will pay attention to:

- The dimension of feature matrix
- The dimension of label or prediction matrix
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

>In binary classification the label matrix always has ``mx1`` dimension. Where ``m`` is the number of samples. During training the labels have values of either 0 or 1. During prediction the values range from 0 to 1. We can round the values to 0 and 1 to get an yes or no answer.

## Create the Model
Add this code.

```python
optimizer, accuracy, X, Y, Y_hat = 
    model.build_model(num_features=2)
```

## Do Training and Prediction

Add this code. This is very similar to regression except ``prediction`` is not a real number. It is a ``mx1` matrix of real numbers ranging from 0 to 1.

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

- train_features.csv - Open this file and observe how the indictor coluns Pclass_2, Pclass_3, etc. are created.
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
Add this code. This is practically same as the rain prediction problem. Except we run training for a longer number of epochs.

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

# Fetal Monitoring Complication Prediction
Cardiotocography is used to monitor fetal heartbeat and  uterine contractions during pregnancy. Various metrics are used to predict complications like hypoxia and acidosis.

Prediction falls under three categories:
- Normal: No hypoxia/acidosis
- Suspect: Low probability of hypoxia/acidosis
- Pathologic: High probability of hypoxia/acidosis, requires immediate action

## Downloading Source Data
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
- CLASS - FHR pattern class code (1 to 10) 
- NSP - fetal state class code (1=normal; 2=suspect; 3=pathologic). This is what we need to predict.

# Titanic Survival Prediction
## Downloading Source Data
Raw data was downloaded from:

```
https://www.kaggle.com/c/titanic
```

These are the source files:

- train.csv - Training data
- test.csv - Test data. But the **Survival** column is not included here for some reason.
- gender_submission.csv - Test survival data is here.

Description of the columns:
- PassengerId - Passenger ID. Not useful for classification.
- Survived - Encoded as 0 (died) and 1 (survived).
- Pclass - Class of travel ticket. 1 for first class and so on.
- Name - Name of passenger. Not useful for classification.
- Sex - Gender: male or female.
- Age - Age.
- SibSp - Number of Sibling/Spouse aboard
- Parch - Number of Parent/Child aboard
- Ticket - Ticket number. Not useful for classification.
- Fare - Price paid for ticket.
- Cabin - Cabin number. Not useful for classification.
- Embarked - Port where embarked on ship. Not useful for classification.

## Preparing Data
It was cleaned up and processed using ``prepare_data.py``. Rows with key missing data are simply excluded instead of filled in with some defaults. Gender is encoded as female=1 and male=2.