#%%
import xgboost as xgb
import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix



#%%
df0= pd.read_csv('../CleanedDatasets/2015_2016_cleaned.csv')
df1 = pd.read_csv('../CleanedDatasets/2016_2017_cleaned.csv')
df2 = pd.read_csv('../CleanedDatasets/2017_2018_cleaned.csv')
df3 = pd.read_csv('../CleanedDatasets/2018_2019_cleaned.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df0 = df0.dropna()

df1 = df1.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam'], axis=1)
df2 = df2.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam'],axis=1)
df3 = df3.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam'],axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam'],axis=1)

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})
#%%
frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)

#%%

df3_cotes_raw = pd.read_csv('../DataSets/2018_2019.csv')
df3_cote = pd.DataFrame()
df3_cote['B365H']=df3_cotes_raw['B365H']
df3_cote['B365D']=df3_cotes_raw['B365D']
df3_cote['B365A']=df3_cotes_raw['B365A']

df3_cote = df3_cote.iloc[50:]
df3_cote = df3_cote.reset_index(drop=True)
#%%
df2_cotes_raw = pd.read_csv('../DataSets/2017_2018.csv')
df2_cotes = pd.DataFrame()
df2_cotes['B365H']=df2_cotes_raw['B365H']
df2_cotes['B365D']=df2_cotes_raw['B365D']
df2_cotes['B365A']=df2_cotes_raw['B365A']

df2_cotes = df2_cotes.iloc[50:]
df2_cotes = df2_cotes.reset_index(drop=True)






# %%
X_train = (np.array(df.drop(['FTR'],axis=1)))
y_train = (np.array(df['FTR']))

X_test = (np.array(df2.drop(['FTR'],axis=1)))
y_test = (np.array(df2['FTR']))
# %%
#clf = GradientBoostingClassifier(n_estimators = 100)
clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

plot_confusion_matrix(clf,X_test,y_test)
# %%
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
#%%

param1={'max_depth':4, 'eta':0.1, 'objective': 'multi:softmax', 'num_class':3}
epochs =10

model = xgb.train(param1, train, epochs)
predictions = model.predict(test)
print(accuracy_score(y_test, predictions))

#%%

def simule_annee_pari(df_cote_test,predictions,y_test,basebet=10):
    gain = 0
    for i in range(len(predictions)):
        if predictions[i]==y_test[i]:
            if predictions[i]==2:
                gain+=basebet*(df_cote_test['B365H'].iloc[i])
            if predictions[i]==1:
                gain+=basebet*(df_cote_test['B365D'].iloc[i])
            if predictions[i]==0:
                gain+=basebet*(df_cote_test['B365A'].iloc[i])
    return (gain-len(y_test)*basebet)
    
print(simule_annee_pari(df2_cotes,predictions,y_test))

# %%
