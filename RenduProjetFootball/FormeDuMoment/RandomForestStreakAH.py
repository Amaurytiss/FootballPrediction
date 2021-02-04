#%%
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix

#%%We are loading datasets, with ou without the new feature created to see wether or note it is useful in our predictions
#2015-2016 to 2018-2019
df0 = pd.read_csv('DatasetsStreakAH/2015_2016_streak_AH.csv')
df1 = pd.read_csv('DatasetsStreakAH/2016_2017_streak_AH.csv')
df2 = pd.read_csv('DatasetsStreakAH/2017_2018_streak_AH.csv')
df3 = pd.read_csv('DatasetsStreakAH/2018_2019_streak_AH.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df0 = df0.dropna()

df1 = df1.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'       ], axis=1)
df2 = df2.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'], axis=1)
df3 = df3.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'], axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'       ], axis=1)

df1_drop = df1.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df2_drop = df2.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df3_drop = df3.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df0_drop = df0.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)
frames_drop = [df0_drop,df1_drop]
df_drop = pd.concat(frames_drop,ignore_index=True)



#%%
df0 = pd.read_csv('DatasetsStreakAH/2017_2018_streak_AH.csv')
df1 = pd.read_csv('DatasetsStreakAH/2018_2019_streak_AH.csv')
df2 = pd.read_csv('DatasetsStreakAH/2019_2020_streak_AH.csv')
df3 = pd.read_csv('DatasetsStreakAH/2020_2021_streak_AH.csv')

df0 = df0[10:]
df1 = df1[10:]
df2 = df2[50:]
df3 = df3[50:]

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df0 = df0.dropna()

df1 = df1.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'       ], axis=1)
df2 = df2.drop(['Time','Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'], axis=1)
df3 = df3.drop(['Time','Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'], axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam','HTF','HTY','HTR','ATF','ATY','ATR'       ], axis=1)

df1_drop = df1.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df2_drop = df2.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df3_drop = df3.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)
df0_drop = df0.drop(['HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'], axis=1)


#,'HomeTeamStreak_AtHome','AwayTeamStreak_AtHome','HomeTeamStreak_AtAway','AwayTeamStreak_AtAway'
#,'HomeTeamStreak','AwayTeamStreak'
#,'HomeTeamStreak','AwayTeamStreak'
#,'HomeTeamStreak','AwayTeamStreak'


#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'
#'HTW','HTD','HTL','ATW','ATD','ATL'

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)
frames_drop = [df0_drop,df1_drop]
df_drop = pd.concat(frames_drop,ignore_index=True)


# %%train and test datasets with and without the new feature

X_train = list(np.array(df.drop(['FTR'],axis=1)))
y_train = list(np.array(df['FTR']))

X_test = list(np.array(df2.drop(['FTR'],axis=1)))
y_test = list(np.array(df2['FTR']))

X_train_drop = list(np.array(df_drop.drop(['FTR'],axis=1)))
y_train_drop = list(np.array(df_drop['FTR']))

X_test_drop = list(np.array(df2_drop.drop(['FTR'],axis=1)))
y_test_drop = list(np.array(df2_drop['FTR']))


# %%Compute 10 forest and compare the scores

for i in range(10):
    clf = RandomForestClassifier(n_estimators=100)
    clf2 = RandomForestClassifier(n_estimators = 100)
    
    clf.fit(X_train,y_train)
    clf2.fit(X_train_drop,y_train_drop)

    print('with new feature :', clf.score(X_test,y_test),'vs without :', clf2.score(X_test_drop,y_test_drop))

#%%
scores_withoutfeature = []
scores_withfeature = []
for i in range(10):
    clf = RandomForestClassifier(n_estimators=100)
    clf2 = RandomForestClassifier(n_estimators = 100)
    
    clf.fit(X_train,y_train)
    clf2.fit(X_train_drop,y_train_drop)

    scores_withfeature.append(clf.score(X_test,y_test))
    scores_withoutfeature.append(clf2.score(X_test_drop,y_test_drop))

print('avg score with new feature :', sum(scores_withfeature)/len(scores_withfeature))
print('avg score without the new feature :', sum(scores_withoutfeature)/len(scores_withoutfeature))


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
