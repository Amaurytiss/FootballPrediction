#%% Tous les import nécéssaires
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from joblib import dump, load
import statistics
#%%Ouverture du dataset, suppression des colonnes de texte, reset des indices
#supression de la première journée où il n'y a encore aucune stats
df = pd.read_csv('../CleanedDatasets/2007_2008.csv')
#%%
df2 = pd.read_csv('../DataSets/2007_2008.csv')
df_cote = pd.DataFrame()
df_cote['B365H']=df2['B365H']
df_cote['B365D']=df2['B365D']
df_cote['B365A']=df2['B365A']
df = df.drop(['Unnamed: 0','Div','Date','HomeTeam','AwayTeam'],axis=1)
df = df.iloc[10:]
df = df.reset_index(drop=True)

df_cote = df_cote.iloc[10:]
df_cote = df_cote.reset_index(drop=True)


#%%prendre les cotes d'un site de pari 


#%%Création des deux dico faisant correspondre équipes et id
#%% nb de matchs pris en compte dans train
nb_train = 300


#%% première random forest sans shuffle
labels_train = np.array(df['FTR'][:nb_train])
data_train = np.array(df.drop(columns='FTR')[:nb_train])

labels_test = np.array(df['FTR'][nb_train:])
data_test = np.array(df.drop(columns='FTR')[nb_train:])


data_train=list(data_train)
labels_train=list(labels_train)
#%%
df_cote_test = df_cote[nb_train:]
#%%
#clf=tree.DecisionTreeClassifier()
clf= RandomForestClassifier(n_estimators=100)
clf=clf.fit(data_train,labels_train)
clf.score(data_test,labels_test)
#%% plusieurs random_forest avec une boucle
best_tree = 0
current_best_score = 0
list_score=[]
for i in range(200):
    labels_train = np.array(df['FTR'][:nb_train])
    data_train = np.array(df.drop(columns='FTR')[:nb_train])

    labels_test = np.array(df['FTR'][nb_train:])
    data_test = np.array(df.drop(columns='FTR')[nb_train:])


    data_train=list(data_train)
    labels_train=list(labels_train)

    #clf=tree.DecisionTreeClassifier()
    clf= RandomForestClassifier(n_estimators=100)
    clf=clf.fit(data_train,labels_train)
    score = clf.score(data_test,labels_test)
    list_score.append(score)
    if score> current_best_score:
        current_best_score=score
        best_tree=clf
        print(current_best_score)
print("average",statistics.mean(list_score))
#%% sauvegarde de l'arbre best tree
dump(best_tree, 'foret_without_teamsV1_100iter_100est_noshuffle_54.joblib')


#%% avec shuffle :
ind=[i for i in range(len(df))]
shuffle(ind)

X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in range(nb_train):
    X_train.append(list(df.drop(['FTR'],axis=1).iloc[ind[i]]))
    y_train.append(df['FTR'].iloc[ind[i]])
for i in range(nb_train, len(df)):
    X_test.append(list(df.drop(['FTR'],axis=1).iloc[ind[i]]))
    y_test.append(df['FTR'].iloc[ind[i]])

#clf=tree.DecisionTreeClassifier()
clf =  RandomForestClassifier()
clf=clf.fit(X_train,y_train)

#%%
best_tree_shuffle = 0
current_best_score = 0
for i in range(10):
    ind=[i for i in range(len(df))]
    shuffle(ind)
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    for i in range(nb_train):
        X_train.append(list(df.drop(['FTR'],axis=1).iloc[ind[i]]))
        y_train.append(df['FTR'].iloc[ind[i]])
    for i in range(nb_train, len(df)):
        X_test.append(list(df.drop(['FTR'],axis=1).iloc[ind[i]]))
        y_test.append(df['FTR'].iloc[ind[i]])
    clf =  RandomForestClassifier(n_estimators=1000)
    clf=clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    if score> current_best_score:
        current_best_score=score
        best_tree_shuffle=clf
    print(current_best_score)
#%%
res=clf.predict(X_test)
compteur = 0
nb_nuls_predis=0
for i in range(len(X_test)):
    #print(res[i]==labels_test[i])
    if res[i]==y_test[i]:
        compteur+=1
    #print('\n')
print(str(compteur*100/len(X_test))+"%")

#%%
plot_confusion_matrix(clf,X_test,y_test)

#%%
clf.predict_proba(X_test)
clf.decision_path(X_test)
print(clf.decision_path(X_test))
clf.score(X_test,y_test)
#%% ne pas écraser foretV1 !!!
#from joblib import dump, load
#dump(clf, 'foretV2.joblib') 
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

#%%
print(simule_annee_pari(best_tree,df_cote_test,data_test,labels_test))
