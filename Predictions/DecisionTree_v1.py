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
df = df.drop(['Unnamed: 0','Div','Date'],axis=1)
df = df.iloc[10:]
df = df.reset_index(drop=True)

#%%Création des deux dico faisant correspondre équipes et id
id_team = {}
liste_des_clubs = []
for i in range(len(df)):
    if df['HomeTeam'][i] not in liste_des_clubs:
        liste_des_clubs.append(df['HomeTeam'][i])
    if df['AwayTeam'][i] not in liste_des_clubs:
        liste_des_clubs.append(df['AwayTeam'][i])

for i in range(len(liste_des_clubs)):
    id_team[liste_des_clubs[i]]=i+1

name_by_id={}
for keys in id_team.keys():
    name_by_id[id_team[keys]]=keys
#%% modification du dataset pour ne plus avoir de str
def nom_vers_num(dataset):
    for i in range(len(df)):
        df['HomeTeam'][i]=id_team[df['HomeTeam'][i]]
        df['AwayTeam'][i]=id_team[df['AwayTeam'][i]]
#%%application de la fonction en place
nom_vers_num(df)
#%% nb de matchs pris en compte dans train
nb_train = 300


#%% première random forest sans shuffle
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
#%% plusieurs random_forest avec une boucle
best_tree = 0
current_best_score = 0
list_score=[]
for i in range(10):
    labels_train = np.array(df['FTR'][:nb_train])
    data_train = np.array(df.drop(columns='FTR')[:nb_train])

    labels_test = np.array(df['FTR'][nb_train:])
    data_test = np.array(df.drop(columns='FTR')[nb_train:])


    data_train=list(data_train)
    labels_train=list(labels_train)

    #clf=tree.DecisionTreeClassifier()
    clf= RandomForestClassifier(n_estimators=500)
    clf=clf.fit(data_train,labels_train)
    score = clf.score(data_test,labels_test)
    list_score.append(score)
    if score> current_best_score:
        current_best_score=score
        best_tree=clf
        print(current_best_score)
print(statistics.mean(list_score))
#%% sauvegarde de l'arbre best tree
dump(best_tree, 'foretV5_300iter_100est_noshuffle_57.joblib')


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
clf = load('foretV1.joblib')
#%%
clf.score(X_test,y_test)