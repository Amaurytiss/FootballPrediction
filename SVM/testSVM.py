#%%
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm

#%%
from sklearn.preprocessing import LabelEncoder
def add_labels_to_df(df):
    res_df = df
    labelencoder=LabelEncoder()
    AwayTeams_labels = labelencoder.fit_transform(df['AwayTeam'])
    HomeTeams_labels = labelencoder.fit_transform(df['HomeTeam'])

    res_df['HomeTeamEncoder'] = HomeTeams_labels
    res_df['AwayTeamEncoder'] = AwayTeams_labels

    return res_df
#%%

df0 = pd.read_csv('../TeamBudget/DataSetsBudget/2015_2016_Budget.csv')
df1 = pd.read_csv('../TeamBudget/DataSetsBudget/2016_2017_Budget.csv')
df2 = pd.read_csv('../TeamBudget/DataSetsBudget/2017_2018_Budget.csv')
df3 = pd.read_csv('../TeamBudget/DataSetsBudget/2018_2019_Budget.csv')



df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df0 = df0.dropna()

df2_name = df2

df1 = df1.drop(['Unnamed: 0','Div','Date'], axis=1)
df2 = df2.drop(['Unnamed: 0','Div','Date'], axis=1)
df3 = df3.drop(['Unnamed: 0','Div','Date'], axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date'], axis=1)
df2_name = df2_name.drop(['Unnamed: 0','Div','Date','HTF','HTY','HTR','ATF','ATY','ATR','HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'], axis=1)

#,'HTF','HTY','HTR','ATF','ATY','ATR','HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HTF','HTY','HTR','ATF','ATY','ATR','HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HTF','HTY','HTR','ATF','ATY','ATR','HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HTF','HTY','HTR','ATF','ATY','ATR','HATT', 'HMIL', 'HDEF', 'AATT', 'AMIL', 'ADEF'
#,'HB','AB'
#,'HB','AB'
#,'HB','AB'
#,'HB','AB'
#'HomeTeam','AwayTeam',
#'HomeTeam','AwayTeam',
#'HomeTeam','AwayTeam',
#'HomeTeam','AwayTeam',

#,'HomeTeamStreak','AwayTeamStreak'
#,'HomeTeamStreak','AwayTeamStreak'
#,'HomeTeamStreak','AwayTeamStreak'
#,'HomeTeamStreak','AwayTeamStreak'


#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df2_name = df2_name.reset_index(drop = True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)
df = df.reset_index(drop=True)

df.drop(['AwayTeam','HomeTeam'], axis = 1 , inplace=True)
df2.drop(['AwayTeam','HomeTeam'], axis = 1 , inplace=True)

# %%
X_train = list(np.array(df.drop(['FTR'],axis=1)))
y_train = list(np.array(df['FTR']))
X_test = list(np.array( df2.drop(['FTR'],axis=1)))
y_test = list(np.array( df2['FTR']))
#%%
X_train2 = list(np.array(df2.drop(['FTR'],axis=1)))
y_train2 = list(np.array(df2['FTR']))
X_test2 = list(np.array(df3.drop(['FTR'],axis=1)))
y_test2 = list(np.array(df3['FTR']))
# %%
clf = svm.SVC()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#%%
#Weird feature importances for a linear kernel only
#pd.Series(abs(clf.coef_[0]), index=df.columns[1:]).nlargest(30).plot(kind='barh')


#%%
proba_predic = clf.predict_proba(X_test)
#%%
#from joblib import dump, load
#dump(clf, 'SavedForest/RandomForest57.joblib')
#%%
for idx,i in enumerate(y_test):
    win_team = 0
    if y_test[idx]==2:
        win_team=df2_name['HomeTeam'].iloc[idx]
    if y_test[idx]==1:
        win_team = "Draw"
    if y_test[idx]==0:
        win_team=df2_name['AwayTeam'].iloc[idx]
    print(df2_name['HomeTeam'].iloc[idx]+" - "+df2_name['AwayTeam'].iloc[idx]+" : "+str(np.flip(proba_predic[idx]))+" and true result is "+win_team+"\n")
#%%
for i in range(10):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))

#%%
best_tree = 0
current_best_score = 0
for i in range(100):

    #clf=tree.DecisionTreeClassifier()
    clf= RandomForestClassifier(n_estimators=100)
    clf=clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    if score> current_best_score:
        current_best_score=score
        best_tree=clf
        print(current_best_score)
#%%
clf.feature_importances_
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

print(simule_annee_pari(clf, df2_cote,X_test,y_test))
# %%
def simule_annee_pari_draw(clf,df_cote,X_test,y_test,basebet=10):
    gain = 0
    perte = 0
    predictions = clf.predict(X_test)
    nb_paris=0
    for idx, i in enumerate(predictions):
        if i==1:
            if i==y_test[idx]:
                gain+=(basebet)*(df_cote['B365D'].iloc[idx]-1)
            else:
                perte+=basebet
            nb_paris+=1
    return (gain-perte,'mise',nb_paris*basebet)

print(simule_annee_pari_draw(clf,df2_cote,X_test,y_test))
# %%
for i in range(10):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    print(simule_annee_pari_draw(clf,df3_cote,X_test,y_test))
# %%
clf.predict_proba(X_test)