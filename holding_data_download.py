# %%
import os
from urllib.request import urlretrieve
from pandas import bdate_range
import time
import pandas as pd
import numpy as np
# %%


def download_file(star_date, end_date, freq='m'):
    main_url = 'https://www.blackrock.com/us/individual/products/239566/'
    middle_url = 'ishares-iboxx-investment-grade-corporate-bond-etf/1464253357814.ajax?fileType=csv&'
    def date_url(x): return 'fileName=LQD_holdings&dataType=fund&asOfDate={Date}'.format(Date=str(x))
    main_url += middle_url

    working_date = get_date(star_date, end_date, freq=freq)

    for date in working_date:
        download_url = main_url + date_url(date)
        urlretrieve(download_url, './DB/holding{date}.csv'.format(date=date))
        time.sleep(0.5)


# %%
def get_date(star, end, freq='m'):
    total_day = bdate_range(star, end)
    df = pd.DataFrame()

    df['o'] = total_day
    df.index = total_day
    df['o'] = df.index.year
    df['p'] = df.index.month
    df['w'] = df.index
    if freq == 'm':
        dd = df.groupby(['o', 'p']).last().w.dt.strftime("%Y%m%d").tolist()
    else:
        dd = df['w'].dt.strftime("%Y%m%d").tolist()
    return dd


# %%
download_file('20180101', '20191231', freq="d")


# %%
