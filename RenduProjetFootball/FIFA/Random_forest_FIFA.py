#%%
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
#%%
df0 = pd.read_csv('DatasetsFIFA/2015_2016_FIFA.csv')
df1 = pd.read_csv('DatasetsFIFA/2016_2017_FIFA.csv')
df2 = pd.read_csv('DatasetsFIFA/2017_2018_FIFA.csv')
df3 = pd.read_csv('DatasetsFIFA/2018_2019_FIFA.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df0 = df0.dropna()

df1 = df1.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'], axis=1)
df2 = df2.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'],axis=1)
df3 = df3.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'],axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'],axis=1)

#,'HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)
# %%
X_train = list(np.array(df.drop(['FTR'],axis=1)))
y_train = list(np.array(df['FTR']))

X_test = list(np.array(df2.drop(['FTR'],axis=1)))
y_test = list(np.array(df2['FTR']))
# %%
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
#%%
#%%Avec le features score d'équipe FIFA EAsports
scores = []
for i in range(1,51):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_test,y_test))
    if i%10==0:
        if i==1:
            print("1ère","forêt :",clf.score(X_test,y_test))
        print(i,"ème forêt :",clf.score(X_test,y_test))
print("avg",sum(scores)/len(scores))
#%%
clf.feature_importances_
plt.bar(x=df.columns[1:],height=clf.feature_importances_,width=0.5,bottom=None, align='center')
plt.xticks(range(len(df.columns[1:])), df.columns[1:], rotation='vertical')
plt.show()
# %%
plot_confusion_matrix(clf,X_test,y_test)
#%%
df2_cotes_raw = pd.read_csv('../DataSets/2017_2018.csv')
df2_cote = pd.DataFrame()
df2_cote['B365H']=df2_cotes_raw['B365H']
df2_cote['B365D']=df2_cotes_raw['B365D']
df2_cote['B365A']=df2_cotes_raw['B365A']

df2_cote = df2_cote.iloc[50:]
df2_cote = df2_cote.reset_index(drop=True)

df3_cotes_raw = pd.read_csv('../DataSets/2018_2019.csv')
df3_cote = pd.DataFrame()
df3_cote['B365H']=df3_cotes_raw['B365H']
df3_cote['B365D']=df3_cotes_raw['B365D']
df3_cote['B365A']=df3_cotes_raw['B365A']

df3_cote = df3_cote.iloc[50:]
df3_cote = df3_cote.reset_index(drop=True)
#%%

def simule_annee_pari(classifier, df_cote_test, X_test, y_test, basebet=10):
    bankroll = 0
    predictions = classifier.predict(X_test)
    for i in range(len(y_test)):
        if predictions[i]==y_test[i]:
            if predictions[i]==2:
                bankroll+=basebet*df_cote_test['B365H'].iloc[i]
            if predictions[i]==1:
                bankroll+=basebet*df_cote_test['B365D'].iloc[i]
            if predictions[i]==0:
                bankroll+=basebet*df_cote_test['B365A'].iloc[i]
        
    return (bankroll-len(y_test)*basebet)

print(simule_annee_pari(clf, df3_cote,X_test,y_test))
# %%
def simule_annee_pari_draw(clf,df_cote,X_test,y_test,basebet=10):
    gain = 0
    perte = 0
    predictions = clf.predict(X_test)
    nb_paris=0
    for idx, i in enumerate(predictions):
        if i==1:
            if i==y_test[idx]:
                gain+=basebet*df_cote['B365D'].iloc[idx]
            else:
                perte+=basebet
            nb_paris+=1
    return (gain-perte,'mise',nb_paris*basebet)

print(simule_annee_pari_draw(clf,df2_cote,X_test,y_test))
# %%
