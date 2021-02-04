#%%
from dico_ratings import score_FIFA
import pandas as pd
import numpy as np
#%% il faut mettre le bon chemin vers les datasets préalablement nettoyés par Gaspard
seasons = ["2015_2016",'2016_2017','2017_2018','2018_2019','2019_2020','2020_2021']
for key in seasons:
    print(key)
    df = pd.read_csv('../cleanedDataset/'+key+'.csv')
    df = df.dropna()
    df = df.drop(['Unnamed: 0'],axis=1)
    df = df.reset_index(drop=True)
    
    df['HATT']=[0]*len(df)
    df['HMIL']=[0]*len(df)
    df['HDEF']=[0]*len(df)
    df['AATT']=[0]*len(df)
    df['AMIL']=[0]*len(df)
    df['ADEF']=[0]*len(df)

    for i in range(len(df)):

        df['HATT'].iloc[i]=score_FIFA[key][df['HomeTeam'].iloc[i]][0]
        df['HMIL'].iloc[i]=score_FIFA[key][df['HomeTeam'].iloc[i]][1]
        df['HDEF'].iloc[i]=score_FIFA[key][df['HomeTeam'].iloc[i]][2]
        df['AATT'].iloc[i]=score_FIFA[key][df['AwayTeam'].iloc[i]][0]
        df['AMIL'].iloc[i]=score_FIFA[key][df['AwayTeam'].iloc[i]][1]
        df['ADEF'].iloc[i]=score_FIFA[key][df['AwayTeam'].iloc[i]][2]
    
    df.to_csv('DatasetsFIFA/'+key+'_FIFA.csv')

    #Enregistre le nouveau dataset dans DatasetsFIFA sous le nom 'saison'_FIFA.csv

# %%
