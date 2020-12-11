import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from FormeAwayHome import add_win_streak_to_dataset
#%%Building the big boy with every features up to date and saving it

#This one already contains Budget and FIFA scores
df0 = pd.read_csv('../TeamBudget/DataSetsBudget/2015_2016_Budget.csv')
df1 = pd.read_csv('../TeamBudget/DataSetsBudget/2016_2017_Budget.csv')
df2 = pd.read_csv('../TeamBudget/DataSetsBudget/2017_2018_Budget.csv')
df3 = pd.read_csv('../TeamBudget/DataSetsBudget/2018_2019_Budget.csv')

df0.drop(['Unnamed: 0'],axis=1,inplace=True)
df1.drop(['Unnamed: 0'],axis=1,inplace=True)
df2.drop(['Unnamed: 0'],axis=1,inplace=True)
df3.drop(['Unnamed: 0'],axis=1,inplace=True)

#adding the streak

df0 = add_win_streak_to_dataset((df0))
df1 = add_win_streak_to_dataset((df1))
df2 = add_win_streak_to_dataset((df2))
df3 = add_win_streak_to_dataset((df3))

#Adding the attendance

df0_public = pd.read_csv('../TeamPublic/DatasetsPublic/2015_2016_with_attendance_avg.csv')
df1_public = pd.read_csv('../TeamPublic/DatasetsPublic/2016_2017_with_attendance_avg.csv')
df2_public = pd.read_csv('../TeamPublic/DatasetsPublic/2017_2018_with_attendance_avg.csv')
df3_public = pd.read_csv('../TeamPublic/DatasetsPublic/2018_2019_with_attendance_avg.csv')


df0['Home attendance'] = df0_public['Home attendance']
df0['Away attendance'] = df0_public['Away attendance']

df1['Home attendance'] = df1_public['Home attendance']
df1['Away attendance'] = df1_public['Away attendance']

df2['Home attendance'] = df2_public['Home attendance']
df2['Away attendance'] = df2_public['Away attendance']

df3['Home attendance'] = df3_public['Home attendance']
df3['Away attendance'] = df3_public['Away attendance']

""" df0.drop(['Unnamed: 0'],axis=1,inplace=True)
df1.drop(['Unnamed: 0'],axis=1,inplace=True)
df2.drop(['Unnamed: 0'],axis=1,inplace=True)
df3.drop(['Unnamed: 0'],axis=1,inplace=True) """
df0.to_csv('DatasetsFeatures/2015_2016_features.csv')
df1.to_csv('DatasetsFeatures/2016_2017_features.csv')
df2.to_csv('DatasetsFeatures/2017_2018_features.csv')
df3.to_csv('DatasetsFeatures/2018_2019_features.csv')