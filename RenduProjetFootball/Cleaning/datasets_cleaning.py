#%% on import les bibliothèques qui nious serons utilses 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
DF = {}
for i in range(2005,2021) : #entre 2005 et 2021
    dataFrame = pd.read_csv("../DataSets/" + str(i) + "_" + str(i+1) + ".csv") 
    indexB365H = list(dataFrame.columns).index("B365H") # Index de la première colone à supprimer. On transforme en liste car on .index() ne s'applique pas au Index. 
    for c in range(indexB365H,len(dataFrame.columns)) : 
        dataFrame = dataFrame.drop(columns=[dataFrame.columns[len(dataFrame.columns)-1]],axis=1)
    
    #pour transformer les lettres pour le full time result
    dataFrame.loc[dataFrame['FTR'] == 'H',['FTR']] = 2
    dataFrame.loc[dataFrame['FTR'] == 'D',['FTR']] = 1
    dataFrame.loc[dataFrame['FTR'] == 'A',['FTR']] = 0
    
    #pour transformer les lettres pour le halftime result
    dataFrame.loc[dataFrame['HTR'] == 'H',['HTR']] = 2
    dataFrame.loc[dataFrame['HTR'] == 'D',['HTR']] = 1
    dataFrame.loc[dataFrame['HTR'] == 'A',['HTR']] = 0
    
    #dataFrame.to_csv(str(i) + "_" + str(i+1) +".csv") # pour regler le problème de type des colonnes FTR et HTR
    #df = pd.read_csv(str(i) + "_" + str(i+1) +".csv")

    #dataFrame = dataFrame.drop(columns = ['Unnamed: 0'])
    DF[str(i) + "_" + str(i+1)] = dataFrame
    print("nombre de colonnes saisons ", i,"/",i+1, ":", len(dataFrame.columns))

#%%
seasons = ["2007_2008","2008_2009","2009_2010","2010_2011","2011_2012","2012_2013","2013_2014","2014_2015","2015_2016","2016_2017","2017_2018","2018_2019","2019_2020","2020_2021"]
for season in seasons:
    df = DF[season]
    new_df = df.copy()

    #On initialise les listes qui vont comprendre les différentes valeurs pour chaque match et pour chaque équipe
    #on les met dans des listes pour créer les colonnes à la fin
    HomeTeamShootslist = []
    HomeTeamShootOnTargetlist = []
    HomeTeamWinlist = []
    HomeTeamDrawlist = []
    HomeTeamLooselist = []
    HomeTeamFoulslist = []
    HomeTeamYellowCardslist = []
    HomeTeamRedCardslist = []
    HomeTeamGoalslist = []
    AwayTeamShootslist = []
    AwayTeamShootOnTargetlist = []
    AwayTeamWinlist = []
    AwayTeamDrawlist = []
    AwayTeamLooselist = []
    AwayTeamFoulslist = []
    AwayTeamYellowCardslist = []
    AwayTeamRedCardslist = []
    AwayTeamGoalslist = []
        
    for i in range(len(new_df)): #on va balayer tout le dataset pour modifier tous les matchs
        Match = new_df.iloc[i] #on enregistre la ligne
        previousMatchDf = new_df.iloc[0:i] #on enregistre tous les match effectué précédemment
        HomeTeam = Match["HomeTeam"] #on enregistre l'équipe domicile et exterieur
        AwayTeam = Match["AwayTeam"]
        HomeTeamMatch = previousMatchDf.loc[(previousMatchDf["HomeTeam"] == HomeTeam) | (previousMatchDf["AwayTeam"] == HomeTeam)] #on enregistre les matchs effectués par l'équipe à domicile et l'équipe à l'exterieur
        AwayTeamMatch = previousMatchDf.loc[(previousMatchDf["HomeTeam"] == AwayTeam) | (previousMatchDf["AwayTeam"] == AwayTeam)]

        #Nombre de match effectué par les équipes
        MatchMadeHT = len(HomeTeamMatch)
        MatchMadeAT = len(AwayTeamMatch)
        
        #boucle pour étudier les données de l'équipe à domicile. On a des v araibles moyennes en générales
        HomeTeamGoals = 0
        HomeTeamShoots = 0
        HomeTeamShootOnTarget = 0
        HomeTeamWin = 0
        HomeTeamDraw =0
        HomeTeamLoose = 0
        HomeTeamFouls = 0
        HomeTeamYellowCards = 0
        HomeTeamRedCards = 0
        
        for index, row in HomeTeamMatch.iterrows(): #pour tous les matchs précédent à domicile ou à l'exterieur de l'équipe HomeTeam
            if(row["HomeTeam"] == HomeTeam): # Match à domicile de HomeTeam
                HomeTeamGoals += row["FTHG"]/MatchMadeHT #on incrémente les différentes variables
                HomeTeamShoots += row['HS']/MatchMadeHT
                HomeTeamShootOnTarget += row['HST']/MatchMadeHT
                HomeTeamFouls += row['HF']/MatchMadeHT
                HomeTeamYellowCards += row['HY']/MatchMadeHT
                HomeTeamRedCards += row['HR']/MatchMadeHT
                if(row['FTHG']>row['FTAG']): #on cherche si elle a gagné perdu ou fait égalité
                    HomeTeamWin += 1
                if(row['FTHG'] == row['FTAG']):
                    HomeTeamDraw += 1
                if(row['FTHG']<row['FTAG']): 
                    HomeTeamLoose +=1
                    
            else : #Match à l'exterieur de HomeTeam
                HomeTeamGoals += row["FTAG"]/MatchMadeHT #on incrémente les différentes varaibles
                HomeTeamShoots += row['AS']/MatchMadeHT
                HomeTeamShootOnTarget += row['AST']/MatchMadeHT
                HomeTeamFouls += row['AF']/MatchMadeHT
                HomeTeamYellowCards += row['AY']/MatchMadeHT
                HomeTeamRedCards += row['AR']/MatchMadeHT
                if(row['FTAG']>row['FTHG']):#on cherche si elle a gagné perdu ou fait égalité
                    HomeTeamWin += 1
                if(row['FTAG'] == row['FTAG']):
                    HomeTeamDraw += 1
                if(row['FTAG']<row['FTHG']): 
                    HomeTeamLoose +=1
        
        
        
        #boucle chercher données de l'équipe à l'exterieur
        AwayTeamGoals = 0
        AwayTeamShoots = 0
        AwayTeamWin = 0
        AwayTeamDraw = 0
        AwayTeamLoose = 0
        AwayTeamShootOnTarget = 0
        AwayTeamFouls = 0
        AwayTeamYellowCards = 0
        AwayTeamRedCards = 0
        
        for index, row in AwayTeamMatch.iterrows():  #pour tous les matchs précédent à domicile ou à l'exterieur de l'équipe AwayTeam
            if(row["HomeTeam"] == AwayTeam): #Match à domicile de AwayTeam
                AwayTeamGoals += row["FTHG"]/MatchMadeAT#on incrémente les différentes varaibles
                AwayTeamShoots += row["HS"]/MatchMadeAT
                AwayTeamShootOnTarget += row['HST']/MatchMadeAT
                AwayTeamFouls += row['HF']/MatchMadeAT
                AwayTeamYellowCards += row['HY']/MatchMadeAT
                AwayTeamRedCards += row['HR']/MatchMadeAT
                if(row['FTHG']>row['FTAG']): #on cherche si elle a gagné perdu ou fait égalité
                    AwayTeamWin += 1
                if(row['FTHG'] == row['FTAG']):
                    AwayTeamDraw += 1
                if(row['FTHG']<row['FTAG']): 
                    AwayTeamLoose +=1
                
            else : #Match à l'exterieur de AwayTeam
                AwayTeamGoals += row["FTAG"]/MatchMadeAT#on incrémente les différentes varaibles
                AwayTeamShoots += row["AS"]/MatchMadeAT
                AwayTeamShootOnTarget += row['AST']/MatchMadeAT
                AwayTeamFouls += row['AF']/MatchMadeAT
                AwayTeamYellowCards += row['AY']/MatchMadeAT
                AwayTeamRedCards += row['AR']
                if(row['FTAG']>row['FTHG']):#on cherche si elle a gagné perdu ou fait égalité
                    AwayTeamWin += 1
                if(row['FTAG'] == row['FTHG']):
                    AwayTeamDraw += 1
                if(row['FTAG']<row['FTHG']) : 
                    AwayTeamLoose +=1
                    
        #on rajoute les variables aux différentes listes            
        HomeTeamShootslist.append(HomeTeamShoots)
        HomeTeamShootOnTargetlist.append(HomeTeamShootOnTarget)
        if MatchMadeHT==0:
            HomeTeamWinlist.append(HomeTeamWin)
            HomeTeamDrawlist.append(HomeTeamDraw)
            HomeTeamLooselist.append(HomeTeamLoose)
        else:
            HomeTeamWinlist.append(HomeTeamWin/MatchMadeHT)
            HomeTeamDrawlist.append(HomeTeamDraw/MatchMadeHT)
            HomeTeamLooselist.append(HomeTeamLoose/MatchMadeHT)
        HomeTeamFoulslist.append(HomeTeamFouls)
        HomeTeamYellowCardslist.append(HomeTeamYellowCards)
        HomeTeamRedCardslist.append(HomeTeamRedCards) 
        HomeTeamGoalslist.append(HomeTeamGoals)
        AwayTeamShootslist.append(AwayTeamShoots)
        AwayTeamShootOnTargetlist.append(AwayTeamShootOnTarget)
        if MatchMadeAT==0:
            AwayTeamWinlist.append(AwayTeamWin)
            AwayTeamDrawlist.append(AwayTeamDraw)
            AwayTeamLooselist.append(AwayTeamLoose)
        else:
            AwayTeamWinlist.append(AwayTeamWin/MatchMadeAT)
            AwayTeamDrawlist.append(AwayTeamDraw/MatchMadeAT)
            AwayTeamLooselist.append(AwayTeamLoose/MatchMadeAT)
        AwayTeamFoulslist.append(AwayTeamFouls)
        AwayTeamYellowCardslist.append(AwayTeamYellowCards)
        AwayTeamRedCardslist.append(AwayTeamRedCards)
        AwayTeamGoalslist.append(AwayTeamGoals)

    print(new_df.columns)
    #On va maintenant enlever les données qui ne serve pas
    indexFTHG = list(new_df.columns).index("FTHG") # Index de la première colone à supprimer. On transforme en liste car on .index() ne s'applique pas au Index.
    dropIndex = indexFTHG #index pour supprimer les colonnes inutiles. 
    for c in range(indexFTHG,len(new_df.columns)): #on enlève toutes les dernières colonnes
        if(new_df.columns[dropIndex]!='FTR'):
            new_df = new_df.drop(new_df.columns[dropIndex],axis=1)
        else :
            dropIndex +=1

    #On va maintenant rajouter les colonnes à notre dataset
    new_df['HTS'] = HomeTeamShootslist
    new_df['HTST'] = HomeTeamShootOnTargetlist
    new_df['HTW'] = HomeTeamWinlist
    new_df['HTD'] = HomeTeamDrawlist
    new_df['HTL'] = HomeTeamLooselist
    new_df['HTF'] = HomeTeamFoulslist
    new_df['HTY'] = HomeTeamYellowCardslist
    new_df['HTR'] = HomeTeamRedCardslist
    new_df['HTG'] = HomeTeamGoalslist
    new_df['ATS'] = AwayTeamShootslist
    new_df['ATST'] = AwayTeamShootOnTargetlist
    new_df['ATW'] = AwayTeamWinlist
    new_df['ATD'] = AwayTeamDrawlist
    new_df['ATL'] = AwayTeamLooselist
    new_df['ATF'] = AwayTeamFoulslist
    new_df['ATY'] = AwayTeamYellowCardslist
    new_df['ATR'] = AwayTeamRedCardslist
    new_df['ATG'] = AwayTeamGoalslist
    new_df.to_csv('Cleaning/'+season+"_cleaned.csv")