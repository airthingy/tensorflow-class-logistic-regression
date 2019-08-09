import pandas as pd

read_training_cols = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare"]
train_data = pd.read_csv('train.csv', usecols=read_training_cols)
#Filter out rows missing age
train_data = train_data[train_data.Age.notnull()]

#Map male and female to numeric value. Female=1, Male=2
train_data = train_data.replace({"Sex": {"female":1, "male":2}})

train_labels = train_data.pop("Survived")

#Save processed data
train_data.to_csv("train_features.csv", header=True, index=False)
train_labels.to_csv("train_labels.csv", header=True, index=False)


#Work with test data. The syntax is almost same as test data. Except
#Survived column is separated out in gender_submission.csv

read_test_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare"]
test_data = pd.read_csv('test.csv', usecols=read_test_cols)
survival_data = pd.read_csv('gender_submission.csv')
#Add the Survived column to test_data. We need to do that before we filter out bad rows 
#that have missing data
test_data["Survived"] = survival_data["Survived"]
#Filter out rows missing age
test_data = test_data[test_data.Age.notnull()]
#Map male and female to numeric value. Female=1, Male=2
test_data = test_data.replace({"Sex": {"female":1, "male":2}})

test_labels = test_data.pop("Survived")

#Save processed data
test_data.to_csv("test_features.csv", header=True, index=False)
test_labels.to_csv("test_labels.csv", header=True, index=False)

