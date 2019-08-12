Here we solve classification problems. We use both linear regression and neural network.

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