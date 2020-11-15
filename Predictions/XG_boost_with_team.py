#%%
import xgboost as xgb
import pandas as pd 
from sklearn.metrics import accuracy_score

#%%
df = pd.read_csv('../CleanedDatasets/2015_2016_cleaned.csv')
df = df.drop(['Unnamed: 0','Div','Date'],axis=1)
df = df.iloc[10:]
df = df.reset_index(drop=True)
df = df.dropna()

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
y_train = np.array(df['FTR'][:nb_train])
X_train = np.array(df.drop(columns='FTR')[:nb_train])

y_test = np.array(df['FTR'][nb_train:])
X_test = np.array(df.drop(columns='FTR')[nb_train:])

#%%
train =xgb.DMatrix(X_train, label=y_train)
test =xgb.DMatrix(X_test, label=y_test)
#%%
param={'max_depth':2, 'eta':0.3, 'objective': 'multi:softmax', 'num_class':3}
epochs =5
#%%
model = xgb.train(param, train, epochs)
#%%
predictions = model.predict(test)
#%%
print(accuracy_score(y_test, predictions))
# %%

# %%
