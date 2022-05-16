# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
from pyfinance.ols import PandasRollingOLS
import matplotlib.pyplot as plt
# %%
sns.set()
# %%

ruter_data = pd.read_csv('ruter_data.csv', parse_dates=['Date']).dropna(how='all', subset=['Bid Price', 'Ask Price'])
ruter_data['Date'] = ruter_data.Date.dt.tz_localize(None)
ruter_data.rename(columns={'Instrument': 'ISIN'}, inplace=True)
ruter_data.sort_values(['ISIN', "Date"], inplace=True)
# %%
ruter_data['delta_prcie'] = ruter_data.groupby('ISIN')['Ask Price'].diff()
# %%
liquidity_data = (ruter_data[['ISIN', 'Date', 'delta_prcie']]
                  .assign(
    month=ruter_data['Date'].dt.month,
    year=ruter_data['Date'].dt.year,
    delta_prcie_plus=lambda x: x.groupby(['ISIN', 'year', 'month'])['delta_prcie'].shift(-1))
    .groupby(['year', 'month', 'ISIN'])[['delta_prcie', 'delta_prcie_plus']]
    .cov()
    .iloc[1::2, 0]
    .reset_index()
    .assign(Date=lambda x: x[['year', 'month']].astype(int).astype(str).eval('year+month'))
    .loc[:, ['ISIN', "delta_prcie", "Date"]]
    .rename(columns={"delta_prcie": 'liquidity'}))


# %%
b_data = pd.read_csv('LQDp.csv', parse_dates=['Date'])
b_data['Rmt'] = np.log(b_data['Adj Close']) - np.log(b_data['Adj Close'].shift(1))
# %%
ruter_data = ruter_data.merge(b_data[['Date', 'Rmt']], how='left', on='Date')
ruter_data['return'] = ruter_data.groupby('ISIN')['Ask Price'].apply(lambda x: np.log(x)-np.log(x.shift(1)))
# %%
b_data = ruter_data[['Date', 'Rmt']].groupby('Date').last().query("index > '2006-01-01'").reset_index()
# %%


def beta_gen():
    for code, table in ruter_data.groupby('ISIN'):
        if len(table) < 1000:
            continue
        try:
            model = PandasRollingOLS(table['return'], table['Rmt'], 144)
            temp_df = table.assign(beta=model.beta)[['Date', 'ISIN', 'beta']].copy()
        except:
            print(code)
            break
        yield temp_df


def downside_gen():
    for code, table in ruter_data.groupby('ISIN'):
        if len(table) < 1000:
            continue
        temp_df = table.assign(DRF=table['return'].rolling(144).quantile(0.05))[['Date', 'ISIN', 'DRF']].copy()
        yield temp_df


# %%
beta_data = pd.concat(beta_gen())
# beta_data = beta_data.groupby('ISIN').apply(lambda df:
#    df.set_index('Date').resample('M').last()
#    )

# %%
Downside_data = pd.concat(downside_gen()).reset_index(drop=True)
Downside_data.set_index('Date').resample('M').last()

# %%
holding_data = (pd.read_csv('2018_2019_data.csv', parse_dates=['Date'])
                .loc[lambda df: df['Asset Class'] != 'Cash']
                .reset_index(drop=True)
                )

# %%
holding_data = holding_data.merge(Downside_data, how='left', on=['Date', 'ISIN'])
# %%
holding_data['weight'] /= 100
new_index = (holding_data.groupby('Date')
             .apply(lambda df: df.eval('weight*Price').sum())
             )
# %%
holding_data['Return'] = (holding_data.groupby("ISIN")['Price'].apply(lambda df: df.pipe(np.log).diff()))
# %%
holding_data.drop(['Name', 'Sector'], axis=1, inplace=True)
# %%


def long_shot_return(df):
    comp_num = len(df)
    quantile = int(comp_num/10)
    factor_return = df['Return'].head(quantile).sum() - df['Return'].tail(quantile).sum()
    return factor_return


def credit_func(df):
    return sum(df.query('Rating == "AA+"').Return - df.query('Rating == "BB"').Return)


# %%
DRF_factor_return = (holding_data
                     .dropna()
                     .groupby('Date')
                     .apply(lambda df: df.sort_values("DRF", ascending=False).pipe(long_shot_return)))
# %%
Size_factor_return = (holding_data
                      .groupby('Date')
                      .apply(lambda df: df.sort_values("MV").pipe(long_shot_return)))
# %%
Duration_return = (holding_data
                   .groupby('Date')
                   .apply(lambda df: df.sort_values("Duration").pipe(long_shot_return)))


# %%
dd = (pd.read_csv(r'D:\_LQD\新LQD\df_factorData_all.csv', usecols=['Date', "ISIN", 'Rating'], parse_dates=['Date'])
      .groupby('ISIN')
      .apply(lambda df: df.dropna().head(1)))
# %%
holding_data = holding_data.merge(dd[['ISIN', 'Rating']], how='left', on=['ISIN'])
# %%
#holding_data.drop("Rating", axis=1, inplace=True)
# %%
credits_return = (holding_data
                  .groupby('Date')
                  .apply(credit_func))
# %%
# %%
Factor_return_table = pd.DataFrame().assign(DRF=DRF_factor_return,
                                            Size=Size_factor_return,
                                            Credit=credits_return)

# %%
Factor_return_table.cumsum()['Credit'].plot(figsize=(20, 6))
plt.title("Factor return")
