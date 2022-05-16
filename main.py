#%%
from statsmodels.api import add_constant
from sklearn.preprocessing import scale
from function_v2 import * 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd 
import numpy as np 
import os
import time
from joblib import Parallel, delayed
import multiprocessing
#%%
sns.set()
os.chdir(r'D:\pwork\MS_papper\DB')
#%%
def market_preprocess(s):
    max_95 = np.percentile(s, 95)
    cap = np.where(s > max_95, max_95, s)
    return cap

# 平行運算
def applyParallel(Grouped, applyfunc, ax=0):
    job_count = multiprocessing.cpu_count()
    
    if type(Grouped) == pd.core.groupby.generic.DataFrameGroupBy:
        rst = Parallel(n_jobs=job_count)(delayed(applyfunc)(group) for name, group in Grouped)

    elif type(Grouped) ==zip:
        rst = Parallel(n_jobs=job_count)(delayed(applyfunc)(a[1], b[1]) for a, b in Grouped)

    if ax==0:
        rst = pd.concat(rst, axis=1).T.assign(Date=Grouped.groups.keys()).set_index('Date')
    else :
        rst = pd.concat(rst)
    return rst

# %%
Gics_name = [
    'Energy', 'Materials', 'Industrials',
    'Consumer Discretionary', 'Consumer Staples', 'Health Care',
    'Financials', 'Information Technology', 'Communication Services',
    'Utilities', 'Real Estate']

#%%
#資料載入
useless_func = lambda x: x not in ['Dirty_price', 'Acc_Int', 'BID', 'YTM',  'Tot_Ret_Ind', 'Sread_BM', 'Sread_Swap', 'Clean_price','COUPON','COUPONS PER YEAR']
bond_data = pd.read_csv('./bond_data/Row_data.csv',
                        engine='c',
                        parse_dates=['Date'],
                        usecols=useless_func)

fundmental_data = pd.read_csv('./fundmental_data/report_data.csv',
                                parse_dates=['date']).rename(columns={'date':'Date'})

Info_data = pd.read_csv('./fundmental_data/estu_ultimatly.csv')
risk_free_data = pd.read_csv('10y.csv',parse_dates=['date']).dropna()
risk_free_data.columns = ['Date', 'rat']

risk_free_data.loc[:, 'rat'] /= 25200
#%%
#bond_data['DRF'] = -bond_data['DRF']

# %%
#資料前處理
bond_data = (bond_data
                .merge(risk_free_data,how='left', on='Date')
                .assign(exret=lambda df: df['return']-df['rat'])
                .merge(Info_data[['ISIN', 'IssuerCUISP']], how='left', on='ISIN')
                .rename(columns= {"IssuerCUISP": 'cusip'})
                .merge(fundmental_data, how='left', on=['cusip','Date'])
                .drop(['rat', 'return', 'MOD_Duration', 'mv'], axis=1)
                .rename(columns= {'ind':'industry', 'MV':"Mvalue"})
                .dropna(subset=['industry'])
                .assign(industry=lambda df:df.industry.astype(int))
                )



# %%
#檢查因子種類
info_col = ['ISIN', 'Date', 'Mvalue']
style_factor = ['convexity', 'ILL', 'DRF', 'DTS', 'Quailty', 'Leverage']
Ind_factor = (bond_data.industry
                .head(100000)
                .sort_values()
                .pipe(lambda s: ['industry_'+str(item) for item in s.unique()])#np.char.add('industry_', s.unique().astype(str)))
            )
# %%
#填補缺值:報酬率
bond_data['exret']=(bond_data['exret']
                        .fillna(bond_data.groupby(['Date','cusip'])['exret'].transform('mean'))
                        .fillna(bond_data.groupby(['Date','industry'])['exret'].transform('mean'))
                        .shift(-1)
                    )
bond_data= bond_data.loc[bond_data.exret != -np.inf].reset_index(drop=True)

#%%
#填補缺值:市值
bond_data['Mvalue'] = (bond_data
                        .groupby('ISIN').Mvalue.fillna(method='backfill')
                        .fillna(bond_data.groupby('ISIN').Mvalue.fillna(method='ffill'))
                    )
bond_data['Mvalue'] = bond_data.groupby('Date')['Mvalue'].transform(market_preprocess)
bond_data['Mvalue'] = bond_data['Mvalue'].fillna(bond_data.groupby(['Date', 'cusip'])['Mvalue'].transform('mean'))
bond_data['Mvalue'] = bond_data['Mvalue'].fillna(bond_data.groupby(['Date', 'industry'])['Mvalue'].transform('mean'))


# %%

#帶市值標準化
def standarlize_w(s,table , mvname='Mvalue'):
    naloc = s.isna()
    
    ts = s.copy()
    w = table.loc[-naloc, mvname].values
    v = ts[-naloc].values
    ts[-naloc] = (v-np.mean(v*w/w.sum()))/np.std(v)
    return ts
#前處理函數
def bond_data_preprocess(df,style_factor=style_factor):
    tdf = df.copy() if type(df) != tuple else df[1].copy()
    tdf.loc[:, style_factor] = (
        tdf[style_factor]
        .apply(standarlize_w, table=tdf)
        .pipe(lambda df:scale(outliear(df.values)))
    )
    tdf.loc[:, style_factor] =tdf[style_factor].fillna(
                        tdf
                            .groupby('industry')[style_factor]
                            .transform('mean')
                            .apply(lambda s: s.fillna(s.mean())),
                        )

    return tdf


# %%
#第二次前處理:因子去離群標準化
bond_data = (bond_data
    .query('Date >= "2009-01-01" & Life>=1')
    .groupby('Date')
    .pipe(applyParallel, applyfunc=bond_data_preprocess, ax=1)
    .reset_index(drop=True)
)

#%%
#second_preprocess

no = bond_data.dropna(subset=['exret']).groupby('Date').size()
no_data_day = no[no<=100].index.tolist()
bond_data = (bond_data
                .dropna(subset=['exret'])
                .query("Date not in @no_data_day")
            )
#%%
#bond_data.to_csv('Factor_data.csv', index=False)
bond_data = pd.read_csv('Factor_data.csv', engine='c', sep=',', parse_dates=['Date'])

# %%

info_col = ['ISIN', 'Date', 'Mvalue']
style_factor = ['convexity', 'ILL', 'DRF', 'DTS', 'Quailty', 'Leverage']
Ind_factor = (bond_data.industry
                .head(100000)
                .sort_values()
                .pipe(lambda s: list('industry_'+str(item) for item in np.unique(s)) )
            )

#%%
def residual_func(x, b):

    pred = (x
            .pipe(pd.get_dummies, columns=['industry'])
            .loc[:, style_factor+Ind_factor]
            .pipe(add_constant)
            .to_numpy()
            .dot(b.values.T)
            )
    residual = x.assign(pred=pred).eval('pred-exret')
    return residual

def R_squard_func(df):
    uper = sum(df.eval("Mvalue * residual**2"))
    down = sum(df.eval("Mvalue * exret**2"))
    return 1-uper/down



def factor_return_func(df):
    return Barra_Reg(df,['market']+style_factor, Ind_factor).factor_return()



#%%
#計算Factor return
factor_return = (bond_data
                .pipe(pd.get_dummies, columns=['industry'])
                .assign(industry=bond_data.industry)
                .pipe(add_constant)
                .rename(columns={'exret':'return', 'const':'market'})
                .groupby('Date')
                #.apply(lambda df: Barra_Reg(df,['market']+style_factor, Ind_factor).factor_return())
                .pipe(applyParallel, factor_return_func)
                .rename(columns=dict(zip(Ind_factor, Gics_name)))
            )
#%%
#factor_return.to_csv('factor_return.csv')
#factor_return = pd.read_csv('factor_return.csv', parse_dates=['Date']).set_index('Date')
#%%
#計算殘差、R squared
zip_func = zip(bond_data.groupby('Date'), factor_return.groupby('Date'))
bond_data['residual'] = applyParallel(zip_func, residual_func, ax=1)
R_squared = bond_data.groupby('Date').apply(R_squard_func)

#%%
ax = (factor_return
            .cumsum()
            .loc[:, style_factor]
            .mul(100)
            .plot(title='Style Factor return', figsize=(12, 6))
            )
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Date")
ax.set_ylabel("Return")
#plt.savefig('../pic/fa_return.jpg')

ax = (factor_return
            .cumsum()
            .loc[:, Gics_name[0:6]]
            .mul(100)
            .plot(title='Style Factor return', figsize=(12, 6))
            )
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Date")
ax.set_ylabel("Return")
#plt.savefig('../pic/ind1_fa_return.jpg')


ax = (factor_return
            .cumsum()
            .loc[:, Gics_name[6:12]]
            .mul(100)
            .plot(title='Style Factor return', figsize=(12, 6))
            )
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Date")
ax.set_ylabel("Return")
#plt.savefig('../pic/ind2_fa_return.jpg')

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Date")
ax.set_ylabel("Return")
#plt.savefig('../pic/ind1_fa_return.jpg')

#%%
ax = (R_squared
            .rolling(252)
            .mean()
            .plot(figsize=(12,6), title = 'R-squared 12 Month average')
            )
ax.set_xlabel("Date")
ax.set_ylabel("R-squared")
#plt.savefig('../pic/R-squared.jpg')


# %%
#Barra EWMA Risk
cov_mat = (factor_return
            .pipe(Barra_moving_cov, window=540, half_life=252, dstr=5)
            .moving_cov()
        )

var_mat = (factor_return
            .pipe(Barra_moving_cov, window=540, half_life=252, dstr=5)
            .moving_var()
        )

# %%
#計算獨有風險
bond_data['sp_var'] = (bond_data
                    .groupby('ISIN')['residual']
                    .transform(lambda s:
                        s.rolling(window=120)
                        .apply(lambda s:Barra_sp_risk(s, window=120, half_life=65, dstr=15), raw=True))
)


# %%
def p_risk(df, cov_ma, factor_col=['const']+style_factor+Ind_factor):
    x_ma = df[factor_col].to_numpy()
    hp = df['weight'].T@x_ma

    #forcast portfolio risk
    factor_cov = (hp@cov_ma)@hp.T
    sp_risk = np.sum(df.eval('(weight*sp_var)**2'))
    P_std = np.sqrt(factor_cov+sp_risk)
    return P_std

def dr_uper(df, var_mat, factor_col=['const']+style_factor+Ind_factor):
    x_ma = df[factor_col].to_numpy()
    hp = (df['weight'].T@x_ma).ravel()
    factor_std = np.sqrt(var_mat.to_numpy().ravel())

    #forcast portfolio risk
    factor_var = sum(hp*factor_std)
    return factor_var


# %%
factor_data_group = (
    bond_data
        .pipe(pd.get_dummies, columns=['industry'])
        .pipe(add_constant)
        .assign(weight=lambda df:df.groupby('Date')['Mvalue'].transform(lambda s: s/sum(s)))
        .loc[lambda df:df['Date']>=cov_mat.index[0][0]]
        .groupby('Date')
        )
#計算投組波動度
risk_zip_func = zip(factor_data_group ,cov_mat.groupby('Date'))
port_folio_risk = list(Parallel(n_jobs=8)(delayed(p_risk)(a[1], b[1]) for a, b in risk_zip_func))
#計算DR ratio
var_zip_func = zip(factor_data_group, var_mat.groupby('Date'))
port_folio_std = list(Parallel(n_jobs=8)(delayed(dr_uper)(a[1], b[1]) for a, b in var_zip_func))

# %%
#Bias test 計算 z值
forcast_return = (bond_data
    .assign(weight=lambda df:df.groupby('Date')['Mvalue'].transform(lambda s: s/sum(s)))
    .loc[lambda df:df['Date']>=cov_mat.index[0][0]]
    .groupby('Date')
    .apply(lambda df: df.eval('weight*exret').sum())
    .rolling(22).sum()
    .shift(-22)
    .pipe(pd.DataFrame, columns=['p_return'])
    .assign(p_risk = port_folio_risk, factor_std=port_folio_std)
    #.dropna()
)

btable = forcast_return.dropna().eval('p_return/p_risk').clip(-3,3)
# %%
std_fun = lambda s, i: s[i:i+252:22].head(12).std()
check_fun = lambda s, i: len(s[i:i+252:22])>=12
# %%
#Bias test 計算 B值與檢定結果
#Insample :1224
B_value = np.array([std_fun(btable, i) for i in range(1225,2210) if check_fun(btable, i)])
Con_int = np.sqrt(1/6)
with_confidence = sum((B_value > 1-Con_int) & (B_value < 1+Con_int))/len(B_value)
over_forecast = sum(B_value < 1-Con_int)/len(B_value)
under_forecast = 1-over_forecast-with_confidence
# %%
print(
    "Over forcast:"+str(round(over_forecast, 4)),
    "Withing confidence:"+str(round(with_confidence, 4)),
    'Under forecast:' + str(round(under_forecast, 4)),
    'Test day:' + str(len(B_value))
)

# %%
Rti_var_mat = (factor_return
            .pipe(Barra_moving_cov, window=30, half_life=15, dstr=5)
            .moving_var()
        )

# %%
rirank = var_mat.apply(lambda s: s.rank(),axis=1).groupby('Date')
rerank = (factor_return
            .apply(lambda s: s.rolling(60)
                        .apply(lambda s:(s*exp_weight(np.arange(60 ,0, -1), 15)).sum(),raw=True))
            .query('index >="2011-03-01"')
            .apply(lambda s: s.rank(),axis=1)
            .groupby('Date')
    )
# %%
zip_func = zip(rerank, rirank)
RIT = [np.corrcoef(x[1], y[1])[1,0] for x, y in zip_func]
RIT = pd.Series(RIT, index=var_mat.index)
RIT.plot()
# %%

forcast_return.drop('p_return', axis=1).to_csv("risk_table.csv")
RIT.to_csv('RIT.csv')
