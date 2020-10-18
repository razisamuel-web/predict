# %%
import pandas as pd

pd.options.display.max_rows = 999
# %%
# https://info.tase.co.il/Heb/MarketData/SecuritiesInfo/Pages/SecuritiesInfo.aspx
df = pd.read_csv('stock_symbols_full_list.csv', index_col=None)

# %%
'OPCE' in list(df['Symbol(English)'])

# %%
df_clean = df[
              (~df['Symbol(English)'].str.contains(".B",regex=False)) &
              (~df['Name(Hebrew)'].str.contains("אגח")) &
              (~df['Symbol(English)'].str.contains(".W",regex=False)) &
              (~df['Symbol(English)'].str.contains(".C",regex=False)) &
              (~df['Name(Hebrew)'].str.contains("ממשל")) &
              (~df['Name(Hebrew)'].str.contains("מ.ק.מ."))
              ]



# %%
df_clean = df_clean.rename(columns={'Symbol(English)': 'symbol'})

# %%
df_clean.to_csv('./symbols/symbols.csv')
# %%
df_clean.shape