''' ---- import required Libraries -----'''
import csv
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

''' --- Web Scraper to get car price, name, mile, body style, year on truecar website --- '''
name = []
price = []
mile = []
style = []
year = []

make = ['audi' , 'bmw', 'ferrari', 'jeep', 'maserati']
for i in make:
    page = 1
    while page != 5:
        url = (f"https://www.truecar.com/used-cars-for-sale/listings/{i}//?page={page}")
        response = requests.get(url)
    
        # Web scraping on truecar website(get model,price,mile)
        soup = BeautifulSoup(response.text, 'html.parser')
        for span in soup.find_all('span', attrs={'class':"vehicle-header-make-model text-truncate"}):
            name.append(span.get_text())

        for div in soup.find_all('div', attrs={'class':"heading-3 margin-y-1 font-weight-bold"}):
            price.append(div.get_text())

        for div in soup.find_all('div', attrs={'class':"d-flex w-100 justify-content-between"}):
            mile.append(div.get_text())

        for div in soup.find_all('div', attrs={'data-test':"vehicleCardTrim"}):
            style.append(div.get_text())

        for span in soup.find_all('span', attrs={'class':"vehicle-card-year font-size-1"}):
            year.append(span.get_text())

        page = page + 1

# Convert name list to two model and company lists
model = []
company = []

for i in name:
    company.append(i.split(' ' , 1)[0])
    model.append(i.split(' ' , 1)[1])

# Convert all lists to one list
rows = [list(x) for x in zip(company, model, price, mile, style, year)]

# Create a csv file and write data to it
column = ['company', 'model', 'price', 'mile', 'style', 'year'] 
rows = [list(x) for x in zip(company, model, price, mile, style, year)]
with open('cars_info.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(column) 
    write.writerows(rows) 

# Read dataset 
dataset = pd.read_csv("cars_info.csv")

# data normalization file
dataset['price'] = [i[1:].replace(',', '') for i in dataset['price']]
dataset['price']=dataset['price'].astype(int)
dataset['mile'] = [i.split(' ', 1)[0].replace(',', '') for i in dataset['mile']]
dataset['mile']=dataset['mile'].astype(int)
dataset['style'] = dataset['style'].str.split().str.slice(start=0,stop=3).str.join(' ')

''' ---- Create models for price forecasting --- '''
X = dataset[['company', 'model', 'mile', 'style', 'year']]
y = dataset['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
ohe = OneHotEncoder()
ohe.fit(X[['model','company','style']])

column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['model','company','style']),
                                        remainder = 'passthrough')
lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test,y_pred)


pipe.predict(pd.DataFrame(columns = X_test.columns, data = np.array(['BMW','X5',22000,'xDrive40i',2019]).reshape(1,5)))

# Create functions to specify the model and style of the car
def model(model_input):
    d = dataset[(dataset['company'] == model_input) & (dataset['model'])]
    return (d['model'].unique()) 

def body_style(style_input):
    d = dataset[(dataset['company'] == style_input ) & (dataset['style'])]
    return(d['style'].unique())

''' --- Get car details from user --- '''
# start app
companies = dataset['company'].unique()
print('Welcom to Car price prediction')
print('Enter the details of the car you want to sell.')
print()

# company
print('Choose one of the companies:\n ', companies)
icompony = input('company : ')
print()

# model
print('Choose one of the following models:\n ', model(icompony))
imodel = input('model: ')
print()

# mile
imile = input('How many miles?')
print()

#style
print('Choose one of the following Body style:\n ', body_style(icompony)) 
istyle = input('body style : ')
print()

# year
iyear = input('Choose a year :')

# Price forecast
answer = pipe.predict(pd.DataFrame(columns = X_test.columns, data = np.array([icompony,imodel,imile,istyle,iyear]).reshape(1,5)))
print('Pridiction: %s$ ' % int(answer))