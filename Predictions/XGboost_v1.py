#%%
import xgboost as xgb
import pandas as pd 
from sklearn.metrics import accuracy_score
#%%

df0= pd.read_csv('../CleanedDatasets/2015_2016_cleaned.csv')
df1 = pd.read_csv('../CleanedDatasets/2016_2017_cleaned.csv')
df2 = pd.read_csv('../CleanedDatasets/2017_2018_cleaned.csv')
df3 = pd.read_csv('../CleanedDatasets/2018_2019_cleaned.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[10:]
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
df0
#%%
frames = [df1,df2]
df = pd.concat(frames,ignore_index=True)
#%%
X_train = (np.array(df.drop(['FTR'],axis=1)))
y_train = (np.array(df['FTR']))

X_test = (np.array(df3.drop(['FTR'],axis=1)))
y_test = (np.array(df3['FTR']))
#%%
train =xgb.DMatrix(X_train, label=y_train)
test =xgb.DMatrix(X_test, label=y_test)
#%%
param={'max_depth':4, 'eta':0.3, 'objective': 'multi:softmax', 'num_class':3}
epochs =10
#%%
model = xgb.train(param, train, epochs)
#%%
predictions = model.predict(test)
#%%
print(accuracy_score(y_test, predictions))
# %%

# %%
