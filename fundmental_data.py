#%%
import pandas as pd 
import numpy as np
import os
# %%
os.chdir(r'D:\pwork\MS_papper\DB\fundmental_data')
# %%
data_name_list = [item for item in os.listdir() if '財報' in item]
# %%
code_col = ['gvkey', 'cusip']
code_col_type = {item:'str' for item in code_col}

# %%
data = pd.concat(
    objs= (pd.read_csv(file_name, dtype=code_col_type) for file_name in data_name_list)
)
data = data.drop_duplicates().reset_index(drop=True)
#%%
use_less_col = ['fyearq', 'fqtr' ,'gvkey', 'curcdq','consol','tic','popsrc', 'conm', 'costat', 'cik', 'datafmt', 'indfmt']
data.drop(use_less_col, axis=1,inplace=True)
data.rename(columns={'oancfy':'ocf','cheq':'cash','mkvaltq':'mv', 'ltq':'Debt','teqq':'Equity', 'atq':'Asset', "actq":'ca',"gsector":'ind','dlttq':'LTB'}, inplace=True)
data = data.drop_duplicates().reset_index(drop=True)
# %%

data.eval('''
Quailty = ocf/Asset + cash/lctq
Leverage = Debt/Equity
''',
inplace=True
)



#%%
estu = pd.read_csv(r'D:\pwork\MS_papper\DB\bond_data\estu_final_modify_2.csv', dtype={'IssuerISIn':'str','CUSIP':'str'})
#%%
cuisp =  pd.read_csv('CUSip_ultimatly.csv', dtype={'CUSIP':'str'})
cuisp.columns = ['IssuerISIn', 'IssuerCUISP']
#%%
new_estu = estu.merge(cuisp, how='left',on=['IssuerISIn'])
# %%
#new_estu.to_csv('estu_ultimatly.csv', index=False)

new_estu = pd.read_csv('estu_ultimatly.csv', dtype={'CUSIP':'str','IssuerCUISP':'str'})
# %%
new_estu.loc[-new_estu.IssuerCUISP.isin(data.cusip.unique()),'IssuerCUISP'].unique()
# %%
new_estu.query('IssuerCUISP== "X3724K132"')

# %%
def count_day(qrange, tday):
    return sum((tday<qrange[1]) & (tday>qrange[0]))
#%%
def expand_df(table):
    totalday =  pd.to_datetime(table.datadate, format='%Y%m%d')


    range_list = np.vstack(list(zip(totalday,totalday[1:])))
    not_year_end = range_list[-1,-1].month<12
    if not_year_end:
        year_end = '{year}1231'.format(year=range_list[-1,-1].year)
        year_end = pd.to_datetime(year_end, format='%Y%m%d')
        range_list = np.vstack([range_list, [range_list[-1,-1], year_end]])

    star = pd.to_datetime(range_list[0,0], format='%Y%m%d')
    end = pd.to_datetime(range_list[-1,-1], format='%Y%m%d')
    total_date = pd.date_range(start=star, end=end,freq='B')

    range_len = len(range_list) #if len(range_list)>len(table) else len(range_list) - 1
    
    qday = [sum((total_date<range_list[i,1]) & (total_date>=range_list[i,0])) for i in  range(range_len)]
    if not_year_end:
        temp =( table
                .assign(qday=qday)
                .pipe(lambda df: df.reindex(df.index.repeat(df.qday)))
                .drop('qday',axis=1)
                .copy()
        )
        

    else:
        temp = (table
                    .drop(table.tail(1).index)
                    .assign(qday=qday)
                    .pipe(lambda df: df.reindex(df.index.repeat(df.qday)))
                    .drop('qday',axis=1)
                    .copy()
        )
    len_dif = len(temp)-len(total_date)
    if len_dif<0:
        temp['date'] = total_date.values[:-1]
    else:
        temp['date'] = total_date.values
    return temp
# %%
resulit = (pd.concat((expand_df(dd) for _, dd in data[['datadate','cusip', 'ind','mv','Quailty','Leverage']].groupby('cusip')))
            .reset_index(drop=True)
            .drop('datadate',axis=1)
            )
#%%
resulit.to_csv('report_data.csv', index=False)

# %%
