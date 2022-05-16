#%%
from functools import reduce
from bondfunc import v_bond_convexity, v_ytm_func
from os.path import join
from numba import njit
import pandas as pd
import numpy as np
from  scipy import optimize
import os
from scipy.stats import norm
from sklearn.preprocessing import scale
from joblib import Parallel, delayed
import multiprocessing
#%%
os.chdir(r'D:\pwork\MS_papper\DB\bond_data')
ES_data = pd.read_csv('estu_final_modify.csv', parse_dates=['Issue Date', 'Maturity Date'])
# %%
def ruter_data_preprocess(data):

        bond_name = [item.split(' - ')[0] for item  in data.columns[1: ]]
        data_col = [item.split(' - ')[-1] for item  in data.columns[1: ]]
        data.columns =['Date'] + [b +'-'+ a for a,b in zip(bond_name, data_col)]


        data.set_index('Date', inplace=True)
        data.columns = pd.MultiIndex.from_tuples([(b, a) for a,b in zip(bond_name, data_col)])
        data = (data
                .unstack(0)
                .reset_index()
                .pivot_table(
                        values=0,
                        index=['level_1', 'Date'],
                        columns='level_0',
                )
        )
        #data.columns = list(data.columns)

        return data

# %%
def read_data(fill_path):
        print(fill_path)
        rdata = (pd.read_csv(fill_path, parse_dates=['Name'], usecols=lambda col: "#ERROR" not in col)
                .pipe(ruter_data_preprocess)
                )
        return rdata

#%%
def merge_func(folder_name, info_table):
        p_join = os.path.join
        cdir = p_join(os.getcwd(), folder_name)

        fill_name_list =[p_join(cdir, name) for name in os.listdir(cdir)]
        data = (
            pd.concat(objs= (read_data(data_name) for data_name in fill_name_list))
                .reset_index()
                .rename(columns={'level_1': 'Name'})
                .merge(info_table[['ISIN', 'Name', 'Issue Date', 'Maturity Date']], how='left', on='Name')
                .rename(columns={'Issue Date':'Issue_Date', 'Maturity Date':'Maturity_Date'})
                .query('Date >= Issue_Date and Date <= Maturity_Date')
                .drop(['Name', 'Issue_Date', 'Maturity_Date'], axis=1)
        )  
        data = data[['ISIN']+ [col for col in data.columns if col != 'ISIN']]
        return data

#%%

Data_folder = [name for name in os.listdir() if '.' not in name]
merge_data = (merge_func(name, ES_data) for name in Data_folder)
Data = reduce(lambda x,y: x.merge(y, how='outer',on=['ISIN', 'Date']), list(merge_data))

# %%
Mod_col = ['MODIFIED DURATION_' + item for item in 'xy']
Data['MODIFIED DURATION'] = Data[Mod_col[0]].fillna(Data[Mod_col[1]])
Data.drop(Mod_col, axis=1, inplace=True)
Data.columns = ['ISIN','Date','ASK','BID','MID','Acc_Int','Dirty_price','MV',"YTM",'Tot_Ret_Ind','Sread_BM','Sread_Swap','MOD_Duration']
Data['Clean_price'] = Data.eval('Dirty_price-Acc_Int')


# %%
Data['Life'] = (
        Data
        .merge(ES_data[['ISIN','Maturity Date']], how='left', on='ISIN')
        .assign(Life = lambda df: df['Maturity Date'] - df['Date'])
        .loc[:, 'Life']
        .astype("timedelta64[D]")
        .divide(365)
        )
#%%
Data = (Data.merge(ES_data[['ISIN','COUPON', 'COUPONS PER YEAR']], how='left', on='ISIN')
        #.query('Life >= 1')
        .reset_index(drop=True)
        .drop(['ASK','MID'],axis=1)
        )
#%%
Data['convexity'] = v_bond_convexity(
        price=Data.Dirty_price.values,
        T=Data.Life.values,
        coup=Data['COUPON'].values,
        freq=Data['COUPONS PER YEAR'].values,
        ytm=Data['YTM'].values/100)

#%%
Data.sort_values(by=['Date', 'ISIN'], inplace=True)

# %%
Ytm_na_loc = Data['YTM'].isna() &(-Data['Dirty_price'].isna())
Ytm_na_table = Data.loc[Ytm_na_loc]

# %%
Data.loc[Ytm_na_loc,'YTM'] = v_ytm_func(
        price = Ytm_na_table.Dirty_price.values, 
        par = 100, 
        T = Ytm_na_table.Life.values,
        coup = Ytm_na_table['COUPON'].values,
        freq = Ytm_na_table['COUPONS PER YEAR'].values,
        guess =0.01
)

Data.loc[Ytm_na_loc,'YTM'] = Data.loc[Ytm_na_loc,'YTM']*100



# %%
Data.dropna(subset=['Acc_Int']).groupby('Date').size().plot()

# %%
Data['ILL'] = -(Data
                .groupby('ISIN')
                .Clean_price
                .transform(lambda s: np.log(s).shift(1).rolling(2, min_periods=1).cov())
                )
# %%
Data['return'] =  (Data
                .groupby('ISIN')
                .Tot_Ret_Ind
                .transform(lambda s: np.log(s) - np.log(s).shift(1))
                )
# %%

# %%
def Var(s):
    mean = np.mean(s)
    std = np.std(s)
    var = norm.ppf(0.05, mean, std)
    return var

rolling_Var = lambda s: s.rolling(250).apply(Var)
    
def applyParallel(Grouped, applyfunc):
    job_count = multiprocessing.cpu_count()
    rst = Parallel(n_jobs=job_count)(delayed(applyfunc)(group) for name, group in Grouped)
    rst = pd.concat(rst)
    return rst
# %%
Data['DRF'] = (Data
                .groupby('ISIN')['return']
                .pipe(applyParallel, applyfunc=rolling_Var)
                #.transform(lambda s: s.rolling(250).apply(Var))
                )


# %%
Data['DTS'] = -Data.eval('MOD_Duration*Sread_Swap*0.0001')

# %%
Data.to_csv('Row_data.csv', index=False)


#%%
Data = pd.read_csv('Row_data.csv', engine='c', delimiter=',', parse_dates=['Date'])

#%%

