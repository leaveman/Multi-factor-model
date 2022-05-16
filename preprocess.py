# %%
import pandas as pd
import numpy as np
import numba as nb
from pyfinance.ols import PandasRollingOLS
import os
# %%
os.chdir('../DB')
holding_list = os.listdir('./holding_data/holding_data_month')
# %%
data_list = [
    pd.read_csv('./holding_data/holding_data_month/'+file_name, skiprows=9).assign(Date=file_name.lstrip('holding').rstrip('.csv'))
        for file_name in holding_list if 'csv' in file_name
]

# %%

unuseful_col = ['Exchange', "SEDOL", 'index', 'Notional Value', 'Mod. Duration', 'Yield to Worst (%)',
                'Real Duration', 'Real YTM (%)', 'Effective Date',
                'Accrual Date', 'Yield to Call (%)', 'Market Currency', 'FX Rate', 'Currency']
data = pd.concat(data_list).reset_index().drop(unuseful_col, axis=1)

data.rename(columns={'Weight (%)':'weight'})[['Date','ISIN', 'weight']].to_csv("holding_data.csv", index=False)
# %%
ESTU = data.dropna(subset=['CUSIP']).query('Sector!= "-"').groupby('ISIN').tail(1).loc[:, ['ISIN','Date','CUSIP','Sector']]
ESTU['Date'] = pd.to_datetime(ESTU['Date'], format='%Y%m%d')
#ESTU.columns = ['code']
#ESTU.to_csv('Estu_3.csv')


#%%


#%%
old_estu = pd.read_csv('ESTU_2.csv')

ESTU.loc[-ESTU['ISIN'].isin(old_estu.ISIN)].to_csv('Newcode.csv',index=False)

# %%


data.rename(columns={
    'Market Value': 'MV', 'Weight (%)': 'weight',
    'Coupon (%)': 'cupon', 'YTM (%)': "YTM"},
    inplace=True)

data = data[
    ["Date"] + data.columns[0: len(data.columns)-2].tolist()
].reset_index(drop=True).dropna()
data.to_csv('2018_2019_data.csv', index=False)
# %%
data = pd.read_csv('2018_2019_data.csv', parse_dates=['Date']).sort_values(['Date', 'ISIN'])
# %%
data['delta_prcie'] = data.groupby('ISIN')['Price'].diff()

# %%
liquidity_data = data[['ISIN', 'Date', 'delta_prcie']].query("ISIN != '-'").assign(
    month=data['Date'].dt.month,
    year=data['Date'].dt.year,
    delta_prcie_plus=lambda x: x.groupby(['ISIN', 'year', 'month'])['delta_prcie'].shift(-1)
).groupby(['year', 'month', 'ISIN'])[['delta_prcie', 'delta_prcie_plus']].cov().iloc[1::2, 0]

# %%
liquidity_data = liquidity_data.reset_index().assign(
    Date=lambda x: x[['year', 'month']].astype(str).eval('year+month')
)[
    ['ISIN', "delta_prcie", "Date"]
].rename(columns={"delta_prcie": 'liquidity'})
# %%
data['return'] = data.groupby('ISIN')['Price'].apply(lambda x: np.log(x)-np.log(x.shift(1)))
# %%
b_data = pd.read_csv('LQDp.csv', parse_dates=['Date'])

b_data['Rmt'] = np.log(b_data['Adj Close']) - np.log(b_data['Adj Close'].shift(1))
# %%
data = data.merge(b_data[['Date', 'Rmt']], how='left', on='Date')

# %%


def beta_gen():
    for code, table in data.query('Date > "2018-01-02"').groupby('ISIN'):
        if (len(table) < 144) or (code == "-"):
            continue
        model = PandasRollingOLS(table['return'], table['Rmt'], 144)

        temp_df = table.assign(beta=model.beta)[['Date', 'ISIN', 'beta']].copy()
        yield temp_df


def downside_gen():
    for code, table in data.query('Date > "2018-01-02"').groupby('ISIN'):
        if (len(table) < 144) or (code == "-"):
            continue
        temp_df = table.assign(DRF=table['return'].rolling(144).quantile(0.05))[['Date', 'ISIN', 'DRF']].copy()
        yield temp_df


# %%
beta_data = pd.concat(beta_gen()).groupby('ISIN').apply(lambda df:
                                                        df.set_index('Date').resample('M').last()
                                                        )

# %%
Downside_data = pd.concat(downside_gen()).reset_index(drop=True)
Downside_data.set_index('Date').resample('M').last()


# %%
ruter_data = pd.read_csv('ruter_data.csv', parse_dates=['Date']).dropna(how='all', subset=['Bid Price', 'Ask Price'])
ruter_data['Date'] = ruter_data.Date.dt.tz_localize(None)
ruter_data.rename(columns={'Instrument': 'ISIN'}, inplace=True)
ruter_data.sort_values(['ISIN', "Date"], inplace=True)
# %%
ruter_data['delta_prcie'] = ruter_data.groupby('ISIN')['Ask Price'].diff()
#%%
ruter_data.Date.isna().sum()
# %%
ruter_data.query('Date <="2006-02-01"')
# %%
