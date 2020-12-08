#%% les bons imports
import pandas as pd 
import numpy as np 
#%%Fonction qui rajoute les séries de victoires / défaites aux datasets

def add_win_streak_to_dataset(df):

    df['HomeTeamStreak']=[0]*len(df)
    df['AwayTeamStreak']=[0]*len(df)
    for i in range(0,len(df)):
        HTStrk=0
        ATStrk=0
        HT = df['HomeTeam'].iloc[i]
        AT = df['AwayTeam'].iloc[i]
        
        for j in range(i-1,-1,-1):
            if df['HomeTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==2 and HTStrk>=0:
                    HTStrk+=1
                if df['FTR'].iloc[j]==2 and HTStrk<0:
                    break
            if df['HomeTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==0 and HTStrk>0:
                    break
                if df['FTR'].iloc[j]==0 and HTStrk<=0:
                    HTStrk-=1
            if df['AwayTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==0 and HTStrk>=0:
                    HTStrk+=1
                if df['FTR'].iloc[j]==0 and HTStrk<0:
                    break
            if df['AwayTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==2 and HTStrk>0:
                    break
                if df['FTR'].iloc[j]==2 and HTStrk<=0:
                    HTStrk-=1
        #LHT.append([HT,HTStrk])
        df['HomeTeamStreak'].iloc[i]=HTStrk


    #################AwayTeam#########################

        for j in range(i-1,-1,-1):
            if df['HomeTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==2 and ATStrk>=0:
                    ATStrk+=1
                if df['FTR'].iloc[j]==2 and ATStrk<0:
                    break
            if df['HomeTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==0 and ATStrk>0:
                    break
                if df['FTR'].iloc[j]==0 and ATStrk<=0:
                    ATStrk-=1
            if df['AwayTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==0 and ATStrk>=0:
                    ATStrk+=1
                if df['FTR'].iloc[j]==0 and ATStrk<0:
                    break
            if df['AwayTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==2 and ATStrk>0:
                    break
                if df['FTR'].iloc[j]==2 and ATStrk<=0:
                    ATStrk-=1
        #AHT.append([AT,ATStrk])    
        df['AwayTeamStreak'].iloc[i]=ATStrk
    return df

#%% on parcourt les saisons, on applique la création des streak et on sauve tout dans le dossier DatasetsStreak
seasons = ["2015_2016",'2016_2017','2017_2018','2018_2019','2019_2020','2020_2021']
for key in seasons:
    df = pd.read_csv('../CleanedDatasets/'+key+'_cleaned.csv')
    df.drop(['Unnamed: 0'],axis = 1, inplace = True)
    df = df.dropna()
    df = df.reset_index(drop = True)
    df_strk = add_win_streak_to_dataset(df)
    df_strk.to_csv('DatasetsStreak/'+key+'_streak.csv')
    