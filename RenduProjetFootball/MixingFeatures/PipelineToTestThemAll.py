#%%
#Algorithme principal, qui permet de tester différents classifier avec différentes features


import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from FormeAwayHome import add_win_streak_to_dataset
from sklearn.metrics import plot_confusion_matrix

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

#Création de la pipeline qui selectionne les bonnes colonnes du dataset en fonction des features spécifiées
def basic(X):
    return X[['HTS',
       'HTST', 'HTW', 'HTD', 'HTL', 'HTG','HTF', 'HTY', 'HTR', 'ATS', 'ATST',
       'ATW', 'ATD', 'ATL', 'ATG','ATF', 'ATY', 'ATR']]

def basic_dataset(X):
    return X[['HTS',
       'HTST', 'HTW', 'HTD', 'HTL', 'HTG', 'ATS', 'ATST',
       'ATW', 'ATD', 'ATL', 'ATG']]

def foul(X):
    return X[['HTF', 'HTY', 'HTR','ATF', 'ATY', 'ATR']]

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

def best_features(X):
    return X[['HTS',
       'HTST', 'HTW', 'HTD', 'HTL', 'HTG','HTF', 'HTY', 'HTR', 'ATS', 'ATST',
       'ATW', 'ATD', 'ATL', 'ATG','ATF', 'ATY', 'ATR','HB', 'AB','Home attendance','Away attendance']]
# pipeline to get all tfidf and word count for first column

pipeline_main_dataset = Pipeline([('main_selection', FunctionTransformer(basic_dataset))])
pipeline_foul = Pipeline([('foul_selection', FunctionTransformer(foul))])
pipeline_budget = Pipeline([('budget_selection', FunctionTransformer(budget))])
pipeline_FIFA = Pipeline([('FIFA_selection', FunctionTransformer(fifa))])
pipeline_streak = Pipeline([('streak_selection', FunctionTransformer(streak))])
pipeline_public = Pipeline([('budget_selection', FunctionTransformer(public))])
pipeline_lables = Pipeline([('labels_selection', FunctionTransformer(ftr))])


def create_X_y(datafram,main_data = True, foul=True,budget=False, fifa = False, streak = False, public= False):
    L=[]
    if main_data:
        L.append(('main_data',pipeline_main_dataset))
    if foul:
        L.append(('foul',pipeline_foul))
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


#%%
#Lecture et nettoyage rapide des datasets

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

df1 = df1.reset_index(drop = True)
df2 = df2.reset_index(drop = True)
df3 = df3.reset_index(drop = True)
df0 = df0.reset_index(drop = True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df0,df1]
df = pd.concat(frames,ignore_index=True)

#%%
#Fonctions qui permet d'afficher toutes les combinaisons de feature possibles et leur score
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

dico = {1:'budget',2:'fifa',3:'streak',4:'public'}
dico_f={'[]':'[]'}
parties = partiesliste([1,2,3,4])
parties.pop(0)
for i in parties:
    aux = []
    for j in i:
        aux.append(dico[j])
    dico_f[str(i)]=str(aux)

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
        for _ in range(1,n_forest):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            scores.append(clf.score(X_test,y_test))
        print("score moyen avec : dataset initial +",dico_f[str(L[i])],':',sum(scores)/len(scores))

        partition_score[str(L[i])] = sum(scores)/len(scores)
    
    return partition_score
#%%teste toutes les combinaisons, avec à chaque fois le score moyen sur n forest
n=5
power = teste_tout(df,df2,n)

#%%Meilleure combinaison de features

X_train,y_train = create_X_y(df ,budget=True,public=True)
X_test,y_test   = create_X_y(df2,budget=True,public=True)

#%%
"""premier test du Classifier Random Forest"""
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train,y_train)
print("score :",clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test)
# %%

"""premier test du Classifier Gradient Boosting"""
clf = GradientBoostingClassifier(max_depth=5, n_estimators=100,learning_rate=0.05)
clf.fit(X_train,y_train)
print("score :",clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test)

# %%

"""premier test du Classifier Arbre de décision"""
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
print("score :",clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test)

# %%

"""premier test du Classifier Regression logistique"""
clf = LogisticRegression()
clf.fit(X_train,y_train)
print("score :",clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test)

# %%

"""Premier test du Classifier SVM"""
clf = SVC()
clf.fit(X_train,y_train)
print("score :",clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test)

#%% plot features importances for random forest (!!! certains classifers n'ont pas de feature importances)
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

plt.bar(x=best_features(df).columns,height=clf.feature_importances_,width=0.5,bottom=None, align='center')
plt.xticks(range(len(best_features(df).columns)), best_features(df).columns, rotation='vertical')
plt.show()

#%%Affiche les prédictions de l'algorithme sur une année

prediction = clf.predict(X_test)

for idx in range(len(y_test)):
    win_team = 0
    pred_team = 0
    if y_test[idx]==2:
        win_team=df2['HomeTeam'].iloc[idx]
    if y_test[idx]==1:
        win_team = "Draw"
    if y_test[idx]==0:
        win_team=df2['AwayTeam'].iloc[idx]
    
    if prediction[idx]==0:
        pred_team = df2['AwayTeam'].iloc[idx]
    if prediction[idx]==2:
        pred_team = df2['HomeTeam'].iloc[idx]
    if prediction[idx]==1:
        pred_team = 'Draw'

    print(df2['HomeTeam'].iloc[idx]+" - "+df2['AwayTeam'].iloc[idx]+" | algo predit : "+pred_team+" | le vrai résultat est : "+win_team+"\n")

#%% Affiche les prédictions avec les probabilités de chaque issue
proba_predic = clf.predict_proba(X_test)
for idx,i in enumerate(y_test):
    win_team = 0
    if y_test[idx]==2:
        win_team=df2['HomeTeam'].iloc[idx]
    if y_test[idx]==1:
        win_team = "Draw"
    if y_test[idx]==0:
        win_team=df2['AwayTeam'].iloc[idx]
    aux = np.flip(proba_predic[idx])
    print(df2['HomeTeam'].iloc[idx]+" - "+df2['AwayTeam'].iloc[idx]+" // prédictions: "+df2['HomeTeam'].iloc[idx]+'= '+str(round(aux[0],3))+" | Draw= "+str(round(aux[1],3))+ " | "+df2['AwayTeam'].iloc[idx]+"= "+str(round(aux[2],3))+" // Le vrai résultat est "+win_team+"\n")

# %%Récupère les cotes de l'année 2017-2018, pour le site Bet365

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

#%% Simulation d'années de paris, avec plusieurs forest pour prendre en compte l'aléatoire dans la génération du classifier

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

for i in range(10):
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train,y_train)
    print("Forest n°"+str(i)+" - mise de 10 € sur chaque paris - gain final : "+str(simule_annee_pari(clf, df2_cote,X_test,y_test))+" €") 

# %%
