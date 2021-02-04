#%% les bons imports
import pandas as pd 
import numpy as np 
#%%Fonction qui rajoute les séries de victoires / défaites aux datasets

def add_win_streak_to_dataset(df):

    df['HomeTeamStreak_AtHome']=[0]*len(df)
    df['HomeTeamStreak_AtAway']=[0]*len(df)
    df['AwayTeamStreak_AtHome']=[0]*len(df)
    df['AwayTeamStreak_AtAway']=[0]*len(df)

    HTStrkH=0
    HTStrkA=0
    ATStrkH=0
    ATStrkA=0

    

    for i in range(0,len(df)):

        HTStrkH=0
        HTStrkA=0
        ATStrkH=0
        ATStrkA=0
        

        

        HT = df['HomeTeam'].iloc[i]
        AT = df['AwayTeam'].iloc[i]
        
        for j in range(i-1,-1,-1): #boucle à l'envers dans le dataset

            if df['HomeTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==2 and HTStrkH>=0:
                    HTStrkH+=1
                if df['FTR'].iloc[j]==2 and HTStrkH<0:
                    break
            if df['HomeTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==0 and HTStrkH>0:
                    break
                if df['FTR'].iloc[j]==0 and HTStrkH<=0:
                    HTStrkH-=1
            if df['AwayTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==0 and HTStrkA>=0:
                    HTStrkA+=1
                if df['FTR'].iloc[j]==0 and HTStrkA<0:
                    break
            if df['AwayTeam'].iloc[j]==HT:
                if df['FTR'].iloc[j]==2 and HTStrkA>0:
                    break
                if df['FTR'].iloc[j]==2 and HTStrkA<=0:
                    HTStrkA-=1
        #LHT.append([HT,HTStrk])
        df['HomeTeamStreak_AtHome'].iloc[i]=HTStrkH
        df['HomeTeamStreak_AtAway'].iloc[i]=HTStrkA

        #if HSH:
        #    HTStrkH=0
        #if HSA:
        #    HTStrkA=0
        #if ASH:
        #    ATStrkH=0
        #if ASA:
        #    ATStrkA=0

    #################AwayTeam#########################

        for j in range(i-1,-1,-1):
            if df['HomeTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==2 and ATStrkH>=0:
                    ATStrkH+=1
                if df['FTR'].iloc[j]==2 and ATStrkH<0:
                    break
            if df['HomeTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==0 and ATStrkH>0:
                    break
                if df['FTR'].iloc[j]==0 and ATStrkH<=0:
                    ATStrkH-=1
            if df['AwayTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==0 and ATStrkA>=0:
                    ATStrkA+=1
                if df['FTR'].iloc[j]==0 and ATStrkA<0:
                    break
            if df['AwayTeam'].iloc[j]==AT:
                if df['FTR'].iloc[j]==2 and ATStrkA>0:
                    break
                if df['FTR'].iloc[j]==2 and ATStrkA<=0:
                    ATStrkA-=1
        #AHT.append([AT,ATStrk])    
        df['AwayTeamStreak_AtHome'].iloc[i]=ATStrkH
        df['AwayTeamStreak_AtAway'].iloc[i]=ATStrkA
    return df

#%% on parcourt les saisons, on applique la création des streak et on sauve tout dans le dossier DatasetsStreak
""" seasons = ["2015_2016",'2016_2017','2017_2018','2018_2019','2019_2020','2020_2021']
for key in seasons:
    df = pd.read_csv('../CleanedDatasets/'+key+'_cleaned.csv')
    df.drop(['Unnamed: 0'],axis = 1, inplace = True)
    df = df.dropna()
    df = df.reset_index(drop = True)
    df_strk = add_win_streak_to_dataset(df)
    df_strk.to_csv('DatasetsStreakAH/'+key+'_streak_AH.csv') """
#%%
