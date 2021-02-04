#%%
import pandas as pd 
import datetime
from selenium import webdriver
from time import sleep



#retourne la liste des matchs
def get_matchs(driver):
    driver.get('https://www.enligne.parionssport.fdj.fr/paris-football/france/ligue-1-uber-eats?filtre=22892')
    sleep(1)
    matchs=driver.find_elements_by_class_name('wpsel-desc')
    res = []
    for i in matchs:
        res.append(i.text)
    return res

def cotes_to_float(val):
    res = ""
    for i in range(len(val)):
        if val[i]==',':
            res+="."
        else:
            res+=val[i]
    return float(res)

#Récupère toutes les cotes et renvoi une liste des cotes par matchs
def get_cotes(driver):
    driver.get('https://www.enligne.parionssport.fdj.fr/paris-football/france/ligue-1-uber-eats?filtre=22892')
    sleep(1)
    cotes = driver.find_elements_by_class_name('outcomeButton-data')
    aux = []
    for i in cotes:
        aux.append(i.text)
    res = []
    for i in range(int(len(cotes)/3)):
        res.append([])
        res[i].append(cotes_to_float(aux.pop(0)))
        res[i].append(cotes_to_float(aux.pop(0)))
        res[i].append(cotes_to_float(aux.pop(0)))
    return res

def create_dico(matchs,cotes):
    res = {}
    for i in range(len(matchs)):
        res[matchs[i]]=cotes[i]
    return res



def main():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver=webdriver.Chrome(executable_path='chromedriver')#,chrome_options=options)
    matchs = get_matchs(driver)
    cotes = get_cotes(driver)
    return create_dico(matchs,cotes)


#%%

print(main())
# %%
