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

df1 = df1.drop(['Unnamed: 0','Div','Date'], axis=1)
df2 = df2.drop(['Unnamed: 0','Div','Date'],axis=1)
df3 = df3.drop(['Unnamed: 0','Div','Date'],axis=1)
df0 = df0.drop(['Unnamed: 0','Div','Date'],axis=1)

df1 = df1.reset_index(drop =True)
df2 = df2.reset_index(drop =True)
df3 = df3.reset_index(drop =True)
df0 = df0.reset_index(drop =True)
df0 = df0.astype({'FTR': 'int64'})

frames = [df2]
df = pd.concat(frames,ignore_index=True)

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

liste_des_clubs
len(liste_des_clubs)
#%% modification du dataset pour ne plus avoir de str
def nom_vers_num(dataset):
    for i in range(len(df)):
        df['HomeTeam'][i]=id_team[df['HomeTeam'][i]]
        df['AwayTeam'][i]=id_team[df['AwayTeam'][i]]
#%%application de la fonction en place
nom_vers_num(df)