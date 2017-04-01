import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import numpy as np
import csv

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket'],axis=1)
    return processed_df

titanic_df = pd.read_csv('train.csv')
titanic_mean = titanic_df['Age'].mean(axis=0)
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_mean)
titanic_df = titanic_df.drop(['Cabin'], axis=1)
titanic_df = titanic_df.dropna()
titanic_df.count()

titanic_df['Survived'].mean()
titanic_df.groupby('Pclass').mean()
class_sex_grouping = titanic_df.groupby(['Pclass','Sex']).mean()
class_sex_grouping
class_sex_grouping['Survived'].plot.bar()

group_by_age = pd.cut(titanic_df["Age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
#age_grouping['Survived'].plot.bar()

processed_df = preprocess_titanic_df(titanic_df)  

X_train = processed_df.drop(['Survived'], axis=1).values
y_train = processed_df['Survived'].values

clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit (X_train, y_train)
print(clf_dt.score (X_train ,y_train))

###Load Test Data
test_df = pd.read_csv('test.csv')
titanic_mean = test_df['Age'].mean(axis=0)
test_df['Age'] = test_df['Age'].fillna(titanic_mean)
ticket_mean = test_df['Fare'].mean(axis=0)
test_df['Fare'] = test_df['Fare'].fillna(ticket_mean)
test_df = test_df.drop(['Cabin'], axis=1)
test_df = test_df.dropna()
test_df.count()
ids = np.array(test_df['PassengerId'])
processed_df = preprocess_titanic_df(test_df) 
X_test = processed_df.values
print('Predicting...')
output = clf_dt.predict(X_test).astype(int)

predictions_file = open("predict.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')
