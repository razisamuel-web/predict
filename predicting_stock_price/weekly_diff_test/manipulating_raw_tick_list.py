#%%
import pandas as pd
pd.options.display.max_rows = 999
#%%
# https://info.tase.co.il/Heb/MarketData/SecuritiesInfo/Pages/SecuritiesInfo.aspx
df  = pd.read_excel('stock_symbols_full_list.xlsx')

#%%
from copy import copy
df_new = copy(df[:])
#%%
df.to_csv('stock_symbols_full_list.csv')
#%%
df  = pd.read_csv('stock_symbols_full_list.csv')

#%%
df_clean = df[(~df['Symbol(English)'].str.contains(".B")) &
              (~df['Name(Hebrew)'].str.contains("אגח")) &
              (~df['Symbol(English)'].str.contains(".W")) &
              (~df['Symbol(English)'].str.contains(".C")) &
              (~df['Name(Hebrew)'].str.contains("ממשל")) &
              (~df['Name(Hebrew)'].str.contains("מ.ק.מ.")) ]
#%%
df_clean
#%%
