#%%

from selenium import webdriver
from time import sleep



#retourne la liste des matchs
def get_matchs(driver):

    driver.get('https://paris-sportifs.pmu.fr/pari/competition/169/football/ligue-1-uber-eats%C2%AE')
    sleep(1)
    matchs=driver.find_elements_by_class_name('trow--event--name')
    res = []
    for i in matchs:
        if len(i.text)<100:
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
    driver.get('https://paris-sportifs.pmu.fr/pari/competition/169/football/ligue-1-uber-eats%C2%AE')
    sleep(1)
    cotes = driver.find_elements_by_class_name('hierarchy-outcome-price')
    res = []
    for i in cotes:
        if len(i.text)>0:
            res.append(cotes_to_float(i.text))
    return res

def create_dico(matchs,cotes):
    res = {}
    for i in range(len(matchs)):
        res[matchs[i]]=[]
        res[matchs[i]].append(cotes[3*i])
        res[matchs[i]].append(cotes[3*i+1])
        res[matchs[i]].append(cotes[3*i+2])
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
