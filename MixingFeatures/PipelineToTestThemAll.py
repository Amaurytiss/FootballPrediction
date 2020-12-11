#%%
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from FormeAwayHome import add_win_streak_to_dataset

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_union
from sklearn.preprocessing import FunctionTransformer



def basic_dataset(X):
    return X[['HTS',
       'HTST', 'HTW', 'HTD', 'HTL', 'HTF', 'HTY', 'HTR', 'HTG', 'ATS', 'ATST',
       'ATW', 'ATD', 'ATL', 'ATF', 'ATY', 'ATR', 'ATG']]

def budget(X):
    return X[['HB', 'AB']]

def fifa(X):
    return X[['HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF']]

def streak(X):
    return X[['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway']]

def public(X):
    return X[['Home attendance','Away attendance']]

def ftr(X):
    return X['FTR']
# pipeline to get all tfidf and word count for first column

pipeline_main_dataset = Pipeline([('main_selection', FunctionTransformer(basic_dataset))])
pipeline_budget = Pipeline([('budget_selection', FunctionTransformer(budget))])
pipeline_FIFA = Pipeline([('FIFA_selection', FunctionTransformer(fifa))])
pipeline_streak = Pipeline([('streak_selection', FunctionTransformer(streak))])
pipeline_public = Pipeline([('budget_selection', FunctionTransformer(public))])
pipeline_lables = Pipeline([('labels_selection', FunctionTransformer(ftr))])


def create_X_y(datafram,main_data = True, budget=False, fifa = False, streak = False, public= False):
    L=[]
    if main_data:
        L.append(('main_data',pipeline_main_dataset))
    if budget:
        L.append(('budget',pipeline_budget))
    if fifa:
        L.append(('fifa',pipeline_FIFA))
    if streak:
        L.append(('streak',pipeline_streak))
    if public:
        L.append(('public',pipeline_public))
    final_transformer = FeatureUnion(L)

    return final_transformer.fit_transform(datafram),pipeline_lables.fit_transform(datafram)

def partiesliste(seq):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j <= jmax:
            if (i>>j)&1 == 1:
                s.append(seq[j])
            j += 1
        p.append(s)
        i += 1 
    return p

#%%
df0 = pd.read_csv('DatasetsFeatures/2015_2016_features.csv')
df1 = pd.read_csv('DatasetsFeatures/2016_2017_features.csv')
df2 = pd.read_csv('DatasetsFeatures/2017_2018_features.csv')
df3 = pd.read_csv('DatasetsFeatures/2018_2019_features.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df0 = df0.dropna()
df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)

#%%
def teste_tout(df,df2,n_forest):
    L = partiesliste([1,2,3,4])
    partition_score = {}
    #budget=False, fifa = False, streak = False, public= False
    for i in range(len(L)):
        budget = False
        fifa   = False
        streak = False
        public = False
        if 1 in L[i]:
            budget=True
        if 2 in L[i]:
            fifa=True
        if 3 in L[i]:
            streak=True
        if 4 in L[i]:
            public=True

        X_train,y_train = create_X_y(df ,budget = budget,fifa = fifa,streak = streak,public = public)
        X_test,y_test   = create_X_y(df2,budget = budget,fifa = fifa,streak = streak,public = public)

        scores = []
        for j in range(1,n_forest):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            scores.append(clf.score(X_test,y_test))
        print("avg",L[i],sum(scores)/len(scores))

        partition_score[str(L[i])] = sum(scores)/len(scores)
    
    return partition_score
#%%
power = teste_tout(df,df2,50)
#%%
X_train,y_train = create_X_y(df ,budget=True,fifa=True)
X_test,y_test   = create_X_y(df2,budget=True,fifa=True)
clf = RandomForestClassifier()

scores = []
for i in range(1,21):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_test,y_test))
    if i%10==0:
        if i==1:
            print("1ère","forêt :",clf.score(X_test,y_test))
        print(i,"ème forêt :",clf.score(X_test,y_test))
print("avg",sum(scores)/len(scores))

#%%Full Features

#%%Plot features importance
plt.bar(x=df.columns[1:],height=clf.feature_importances_,width=0.5,bottom=None, align='center')
plt.xticks(range(len(df.columns[1:])), df.columns[1:], rotation='vertical')
plt.show()


#%%


