


import duka.app.app as import_ticks_methods
from duka.core.utils import TimeFrame
import datetime
import pandas as pd

start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 1, 4)

ratio = 'EURUSD'
Assets = [ratio]

import_ticks_methods(Assets, start_date, end_date,1, TimeFrame.D1, ".", True)
# %%
tick_data = pd.read_csv("EURUSD-2019_01_01-2019_01_04.csv")


