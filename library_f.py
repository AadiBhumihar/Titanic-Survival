# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:21:00 2017

@author: bhumihar
"""

import matplotlib.pyplot as plt
%matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
from tensorflow.contrib import skflow

titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])
titanic_df.head()
titanic_df['survived'].mean()
titanic_df.groupby('pclass').mean()
class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping
class_sex_grouping['survived'].plot.bar()
group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()
titanic_df.count()
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
titanic_df = titanic_df.dropna()
titanic_df.count()

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
    return processed_df
    
processed_df = preprocess_titanic_df(titanic_df)
processed_df

X = processed_df.drop(['survived'], axis=1).values
y = processed_df['survived'].values