# %%
from data_generation import get_data
import requests
import pandas as pd
import numpy as np
import datetime


ticker_name = "SEFA:IT"
interval = 'y1'
def extract_ticks(interval=interval):
    url = "https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-chart"
    querystring = {"interval": interval, "id":ticker_name }
    headers = {
        'x-rapidapi-host': "bloomberg-market-and-financial-news.p.rapidapi.com",
        'x-rapidapi-key': "fa8754ae5bmsh32b7198e7277f82p156901jsn871ee921b6ca"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    json_d = response.json()
    return json_d


# Get year to date data ytd
json_d = extract_ticks()
print(json_d)

# %%
import json

with open("SEFA", "w") as fp:
    json.dump(json_d, fp)

# %%
with open("SEFA", "r") as fp:
    json_d = json.load(fp)
ticks_d = json_d['result']['SEFA:IT']['ticks']
df = pd.DataFrame(ticks_d)
df['Close'] = df['close']
df['Date'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df = df.set_index('time')
data = df.sort_index(ascending=True, axis=0)
data

#%%
new_data = data[['Date','Close']]
index = range(0,len(new_data))
new_data['index']=index
new_data=new_data.set_index('index')
new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data

#%%
df_old_raw = get_data(ticker_name='SEFA',data_from_csv= True,path_from='raw_data')

#%%
df_weekly['days_in_week'][-1] < 3