#Récupération des cotes des différents sites de pari
#%%
import datetime
from selenium import webdriver
from time import sleep

#Récupère la liste des matchs à la une, et les stocke dans une liste list_matchs et dans un dico vide

def get_matchs(driver):
    driver.get('https://www.winamax.fr/paris-sportifs/sports/1/7/4')
    sleep(1)
    matchs=driver.find_elements_by_class_name('sc-prPLn')
    dico_matchs={}
    list_matchs=[]
    for i in range(len(matchs)-1):
        dico_matchs[matchs[i].text]=[]
        list_matchs.append(matchs[i].text)
    return dico_matchs,list_matchs

#Récupère toutes les cotes et renvoi la liste d'éléments web brute
def get_cotes_raw(driver):
    #driver=webdriver.Chrome(executable_path='/Users/Utilisateur/Informatique/chromedriver')

    driver.get('https://www.winamax.fr/paris-sportifs/sports/1/7/4')
    #driver.get('https://www.winamax.fr/paris-sportifs/sports/1/7/19')
    sleep(1)
    return (driver.find_elements_by_class_name('sc-fznWqX'))

#Nettoie la liste des cotes en supprimant les valeurs non conformes
def clean_cotes(cotes):
    res=[]
    for i in range(len(cotes)):
        if cotes[i].text[0]!="+" and cotes[i].text[0:3]!= '...' and len(cotes[i].text)<=7:
            if cotes[i].text[1]==',':
                aux1=(cotes[i].text[:4])
                aux2=aux1[0]+'.'+aux1[2:4]
                res.append(float(aux2))
        #if cotes[i].text[0]!="+" and (len(cotes[i].text)==6 or len(cotes[i].text)==4):
            else:
                res.append(float(cotes[i].text[:2]))
    #à faire mieux:
    res=res[:len(res)-2]
    return res

#rempli le dico des matchs avec les cotes correspondantes
def matchs_cotes(dico_matchs,list_matchs,cleaned_cotes):
    c=0
    n=0
    for i in range(len(cleaned_cotes)-1):
        dico_matchs[list_matchs[n]].append(cleaned_cotes[i])
        c+=1
        if c%3==0:
            n+=1
            c=0
    return dico_matchs


def main():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver=webdriver.Chrome(executable_path='chromedriver')#,chrome_options=options)
    dico_matchs , list_matchs = get_matchs(driver)
    cleaned_cotes=clean_cotes(get_cotes_raw(driver))
    dico_matchs=(matchs_cotes(dico_matchs,list_matchs,cleaned_cotes))
    return dico_matchs


#%%

print(main())




        




