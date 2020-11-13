#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
from sklearn import datasets
from sklearn import tree
from random import shuffle
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
#%%Ouverture du dataset, suppression des colonnes de texte

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
#%%
def nom_vers_num(dataset):
    for i in range(len(df)):
        df['HomeTeam'][i]=id_team[df['HomeTeam'][i]]
        df['AwayTeam'][i]=id_team[df['AwayTeam'][i]]
#%%
nom_vers_num(df)
#%%
nb_train = 290

#%% pour le shuffle
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
#%% sans shuffle
labels_train = np.array(df['FTR'][:nb_train])
data_train = np.array(df.drop(columns='FTR')[:nb_train])

labels_test = np.array(df['FTR'][ind[nb_train:]])
data_test = np.array(df.drop(columns='FTR')[nb_train:])


data_train=list(data_train)
labels_train=list(labels_train)

clf=tree.DecisionTreeClassifier()
clf=clf.fit(data_train,labels_train)
#tree.plot_tree(clf)
#%% avec shuffle
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
#tree.plot_tree(clf)
#%% sans shuffle
res=clf.predict(data_test)
compteur = 0
nb_nuls_predis=0
for i in range(len(data_test)):
    #print(res[i]==labels_test[i])
    if res[i]==labels_test[i]:
        compteur+=1
    #print('\n')
print(str(compteur*100/len(data_test))+"%")
#%% avec shuffle
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
plot_confusion_matrix(clf,X_test,res)


