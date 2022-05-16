#%%
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import os
# %%
plt.style.use('ggplot') 
sns.set()
os.chdir(r'D:\pwork\MS_papper\DB')
#%%

def ad_rank_ace(s):
    return (s-s.min())/(s.max()-s.min())

# %%
return_data = pd.read_csv('./bond_data/Row_data.csv',
                        engine='c',
                        parse_dates=['Date'],
                        usecols=['ISIN','Date' ,'Tot_Ret_Ind']
                        )#,'Clean_price', 'COUPON','COUPONS PER YEAR'])
holding_data = pd.read_csv('holding_data.csv', parse_dates=['Date']).dropna().query('ISIN!= "-"')

bond_data = pd.read_csv('Factor_data.csv',
                            engine='c',
                            sep=',',
                            parse_dates=['Date'],
                            usecols=lambda x: x not in ['Mvalue, Life']
                        )
risk_free_data = pd.read_csv('10y.csv',parse_dates=['date']).dropna()
risk_free_data.columns = ['Date', 'rat']

risk_free_data.loc[:, 'rat'] /= 25200
# %%
# Info_data_col = Info_data.columns.tolist()
# Info_data_col[-1] = 'c_per_year'
# Info_data.columns = Info_data_col
# Info_data.loc[:, 'COUPON'] = Info_data.loc[:, 'COUPON'] /100
#Info_data['Effect_yield'] = Info)
#Info_data.eval('eff_coup = (1+ COUPON/c_per_year)**c_per_year -1', inplace=True)
# %%
holding_data['weight'] = holding_data.groupby('Date')['weight'].transform(lambda s:s/sum(s))

return_data['ret'] = (return_data
                        .sort_values(by=['ISIN','Date'])
                        .groupby('ISIN')['Tot_Ret_Ind']
                        .transform(lambda s:np.log(s)-np.log(s.shift(1)))
                        .fillna(0)
                        )
# %%
style_factor = ['convexity', 'ILL', 'DRF', 'DTS', 'Quailty', 'Leverage']
# %%
# bond_data = (bond_data.merge(Info_data, how='left', on=['ISIN', 'Date'])
#                 .assign(weight=lambda df:df.groupby('Date')['Mvalue']
#                 .transform(lambda s: s/sum(s)))
#                 )


#%%
#篩出月末權重
# change_date = (bond_data
#                     .set_index('Date')
#                     .assign(year=lambda df:df.index.year, month= lambda df:df.index.month, Date=lambda df:df.index)
#                     .groupby(['year', "month"]).last()
#                     .Date
#                     .tolist()
#                 )
# change_date.append(pd.to_datetime('2020-03-16'))
# change_date.sort()

# weight_table = (bond_data
#                     .set_index('Date')
#                     .loc[change_date]
#                     .reset_index()
#                     )

weight_table = (holding_data
                        .merge(bond_data, how='left', on=['Date', 'ISIN'])
                        .query('"2020-12-31">Date>="2009-02-05"')
                        #.assign(weight=lambda df:df.groupby('Date')['Mvalue'].transform(lambda s: s/sum(s)))
                        )
change_date = list(weight_table.Date.unique())
#change_date.append(pd.to_datetime('2020-03-16'))
change_date.sort()
factor_weight_col = [item+"_weight" for item in style_factor]
risk_col = [item+"_weight" for item in ['ILL', 'DRF', 'DTS', 'Leverage']]
# %%
def add_rank(table, factor):
    na_loc = table[factor].isna()
    rank=(table
            .loc[-na_loc]
            .sort_values(by=factor)
            .assign(rank=lambda df: np.arange(1, len(df)+1))
            .loc[:, 'rank']
        )


    rank = pd.concat([rank ,table.loc[na_loc, factor]])
    return rank

def adj_weight(temp_table, bender_coef=0.9):
    retain_r = bender_coef
    change_table = temp_table.dropna(subset=['rank']).sort_values(by=['rank'])
    dec_rank, mid_rank, ace_rank =  np.percentile(change_table['rank'],[20,80,100]).astype(int)
    ace_table = change_table.tail(dec_rank).copy()##.query('@mid_rank<rank<=@ace_rank').copy()
    dec_table = change_table.head(dec_rank).copy()#.query('rank<=@dec_rank').copy()

    adj_list = pd.concat((ace_table.ISIN, dec_table.ISIN)).tolist()
    mid_table = temp_table.query('ISIN not in @adj_list').copy()#.query('@dec_rank<rank<=@mid_rank').copy()

    ace_table['rank'] = ad_rank_ace(ace_table['rank'])
    ace_table['weight_adj'] = ace_table.eval('weight*(1-rank)*@retain_r')
    train_weight = ace_table.eval('weight-weight_adj').sum()

    #dec_rank+=1

    dec_table['weight'] += dec_table.eval('@train_weight*(@dec_rank-rank)') / np.arange(1, dec_rank).sum()
    ace_table['weight'] = ace_table['weight_adj']
    useful_col = ['Date', 'ISIN', 'weight']
    temp_table_2 = pd.concat([ace_table[useful_col], mid_table[useful_col], dec_table[useful_col]])
    temp_table_2['weight'] /= sum(temp_table_2['weight'])
    return temp_table_2



#%%
#計算低風險增值投組
factor_weight_list = [weight_table]
for item in style_factor:
    factor_weight_list.append(
                    weight_table
                        .assign(rank=lambda df:df.groupby('Date').apply(add_rank, factor=item).reset_index('Date', drop=True))
                        .groupby('Date')
                        .apply(adj_weight)
                        .reset_index(drop=True)
                        .rename(columns={'weight':item+'_weight'})
    )

#合併

weight_table_low_risk = reduce(lambda x,y: x.merge(y, how='left', on=["Date", 'ISIN']), factor_weight_list)
weight_table_low_risk = (weight_table_low_risk
                            .loc[:, ['Date', 'ISIN','weight']+factor_weight_col]
                            .drop_duplicates(subset=['Date', "ISIN"])
                            .assign(all_factor = lambda df: df.sum(axis=1)/6,
                                    risk_factor = lambda df: df[risk_col].sum(axis=1)/4
                                )
                            )

#weight_table_low_risk.eval('all_factor = ('+ '+'.join(factor_weight_col)+')/6', inplace=True)
weight_table_low_risk_dict = {Date: table for Date, table in weight_table_low_risk.groupby('Date')}

#%%
#計算高風險增值投組
factor_weight_list = [weight_table]
for item in style_factor:
    factor_weight_list.append(
                    weight_table
                        .assign(rank=lambda df:df.groupby('Date')[item].rank())
                        .sort_values(by=['Date', "rank"])
                        .groupby('Date')
                        .apply(adj_weight, bender_coef=0)
                        .reset_index(drop=True)
                        .rename(columns={'weight':item+'_weight'})
        )
weight_table_high_risk = reduce(lambda x,y: x.merge(y, how='left', on=["Date", 'ISIN']), factor_weight_list)
weight_table_high_risk = (weight_table_high_risk
                            .loc[:, ['Date', 'ISIN','weight']+factor_weight_col]
                            .drop_duplicates(subset=['Date', "ISIN"])
                            .assign(all_factor = lambda df: df.sum(axis=1)/6,
                                    risk_factor = lambda df: df[risk_col].sum(axis=1)/4
                                )
                            )
# weight_table_high_risk.eval('all_factor = ('+ '+'.join(factor_weight_col)+')/6'
#                             , inplace=True)
weight_table_high_risk_dict = {Date:table for Date, table in weight_table_high_risk.groupby('Date')}
#%%


# %%
#參數控制
start_date =' "2009-02-27"'
count_col = ['weight'] + factor_weight_col+ ['all_factor', 'risk_factor']
groupy =  return_data.query('Date>'+start_date)[['Date', 'ISIN', 'ret']].groupby('Date')
# %%
#進行風險控制增值
port_return_li = []
current_weight_table = weight_table_low_risk.query('Date==' + start_date)
extrem_time = True

#計算報酬
for Date, table in groupy:
    tt = current_weight_table.merge(table.drop('Date', axis=1), how='left', on='ISIN').dropna()
    #factor_weight_col
    weight_col = tt[count_col].values#/np.sum(tt[count_col].values, axis=0)
    port_return  = tt.ret.values[None, :] @ weight_col
    port_return_li.append(port_return)


    # if str(Date.date())>="2013-05-01":
    #     extrem_time = True
    if Date in change_date:
        if not extrem_time:
            current_weight_table = weight_table_high_risk_dict[Date]
        else:
            current_weight_table = weight_table_low_risk_dict[Date]
            #extrem_time = False


port_return_df_risk_ctr=pd.DataFrame(np.vstack(port_return_li), index=groupy.groups.keys())
port_return_df_risk_ctr.columns = ['benchmark'] + style_factor + ['Muti-factor','risk_factor']

#%%
#無風險控制

port_return_li = []
current_weight_table=weight_table_high_risk.query('Date=='+start_date)
change_time = 0
#計算報酬
for Date, table in groupy:
    tt = current_weight_table.merge(table.drop('Date', axis=1), how='left', on='ISIN').dropna()
    weight_col = tt[count_col].values#/np.sum(tt[count_col].values, axis=0)
    port_return  = tt.ret.values[None, :]@weight_col
    port_return_li.append(port_return)

    if (Date in change_date)&(Date!=pd.to_datetime('2020-03-16')):
        change_time += 1
        current_weight_table = weight_table_high_risk_dict[Date]



port_return_df_risk=pd.DataFrame(np.vstack(port_return_li), index=groupy.groups.keys())
port_return_df_risk.columns = ['benchmark']  + style_factor + ['Muti-factor', 'risk_factor']

# %%

factor_data = pd.read_csv('factor_return.csv', parse_dates=['Date'], index_col="Date")
RIT = pd.read_csv('RIT.csv', parse_dates=['Date'], index_col="Date")
dr_table = pd.read_csv("risk_table.csv", parse_dates=['Date'], index_col="Date")

# %%

#繪圖區-----------------------------------------------------------------

#time_exper = '"2013-11-01">=index>="2012-01-01"'
time_exper_all = '"2020-12-31">index>="2011-03-01"'
#circis 發生期間
time_exper_tap = '"2013-11-01">=index>="2013-04-01"'
time_exper_fed = '"2018-03-31">=index>="2017-12-01"'
time_exper_china = '"2015-07-31">=index>="2015-04-01"'
time_exper_covid19 = '"2020-05-01">=index>="2019-12-01"'


#風險係數觀察期
time_exper_tap_ob = '"2013-11-01">=index>="2012-05-01"'
time_exper_fed_ob = '"2018-03-31">=index>="2017-01-01"'
time_exper_china_ob = '"2015-12-31">=index>="2014-01-01"'
time_exper_covid19_ob = '"2020-05-09">=index>="2019-01-01"'

cricis_rit = RIT.query(time_exper_all)['0']
cricis_dr = dr_table.query(time_exper_all).eval('factor_std/p_risk')
cricis_return = port_return_df_risk_ctr.query(time_exper_all)['benchmark'].cumsum()
#cricis_10y = risk_free_data.set_index('Date').query(time_exper_all)
#.resample('MS', loffset=pd.Timedelta(14, 'd')).last()
#

cricis_tap_period = RIT.query(time_exper_tap).index 
cricis_tap_bgn = cricis_tap_period[0]
cricis_tap_end = cricis_tap_period[-1]


cricis_euro_period = RIT.query('"2012-07-15">=index>="2012-04-01"').index 
cricis_euro_bgn = cricis_euro_period[0]
cricis_euro_end = cricis_euro_period[-1]


cricis_covid19_period = RIT.query(time_exper_covid19).index 
cricis_covid19_bgn = cricis_covid19_period[0]
cricis_covid19_end = cricis_covid19_period[-1]

cricis_fed_period = RIT.query(time_exper_fed).index 
cricis_fed_bgn = cricis_fed_period[0]
cricis_fed_end = cricis_fed_period[-1]


cricis_china_period = RIT.query(time_exper_china).index 
cricis_china_bgn = cricis_china_period[0]
cricis_china_end = cricis_china_period[-1]




#%%
props = dict(boxstyle='round', alpha=0.5)
fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_rit.mul(100).plot(title='Risk Tolerance Ratio', subplots=True, ax=axes)
#axes.axvspan(cricis_euro_bgn, cricis_euro_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_tap_bgn, cricis_tap_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_covid19_bgn, cricis_covid19_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_fed_bgn, cricis_fed_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_china_bgn, cricis_china_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.text(0.21, 1.05, 'Taper tantrum', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.40, 1.05, '2015 cricis', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.63, 1.05, 'Fed shrink ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.83, 1.05, 'Covid-19 ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.savefig('../pic/Risk Tolerance Ratio.jpg'
            ,bbox_inches='tight'
            ,pad_inches=0)
plt.show()





fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_dr.plot(title='Diversification ratio', subplots=True, ax=axes)
#axes.axvspan(cricis_euro_bgn, cricis_euro_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_tap_bgn, cricis_tap_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_covid19_bgn, cricis_covid19_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_fed_bgn, cricis_fed_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_china_bgn, cricis_china_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.text(0.21, 1.05, 'Taper tantrum', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.40, 1.05, '2015 cricis', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.63, 1.05, 'Fed shrink ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.83, 1.05, 'Covid-19 ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig('../pic/Diversification ratio.jpg'            
            ,bbox_inches='tight'
            ,pad_inches=0)
plt.show()

fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_return.mul(100).plot(title='Benchmark return', subplots=True, ax=axes)
#axes.axvspan(cricis_euro_bgn, cricis_euro_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_tap_bgn, cricis_tap_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_covid19_bgn, cricis_covid19_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_fed_bgn, cricis_fed_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.axvspan(cricis_china_bgn, cricis_china_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
axes.text(0.18, 1.05, 'Taper tantrum', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.39, 1.05, '2015 cricis', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.65, 1.05, 'Fed shrink ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.text(0.87, 1.05, 'Covid-19 ', transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
axes.set_xlabel('Date')

axes.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.savefig('../pic/Benchmark return.jpg'
            ,bbox_inches='tight'
            ,pad_inches=0)
plt.show()

# fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
# cricis_10y.plot(title='10y bill', subplots=True, ax=axes)
# axes.axvspan(cricis_tap_bgn, cricis_tap_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
# axes.axvspan(cricis_covid19_bgn, cricis_covid19_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
# axes.axvspan(cricis_fed_bgn, cricis_fed_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
# axes.axvspan(cricis_china_bgn, cricis_china_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
# axes.text(0.21, 1.05, 'Tap tantrum', transform=axes.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
# axes.text(0.40, 1.05, 'China cricis', transform=axes.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
# axes.text(0.63, 1.05, 'Fed shrink ', transform=axes.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
# axes.text(0.83, 1.05, 'Covid-19 ', transform=axes.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
# #plt.savefig('../pic/Benchmark return.jpg')
# plt.show()



#%%
fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_rit.plot(title='Risk Tolerance Ratio', subplots=True, ax=axes)
plt.show()



fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_dr.plot(title='Diversification ratio', subplots=True, ax=axes)
plt.show()

fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
cricis_return.plot(title='Benchmark return', subplots=True, ax=axes)
plt.show()

# fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
# cricis_10y.plot(title='10y bill', subplots=True, ax=axes)
# risk_free_data.set_index('Date').query(time_exper_fed_ob)
# plt.show()
# %%

# %%
time_exper_li = [   '"2013-07-01">=index>="2013-05-01"',
                    '"2018-03-01">=index>="2017-12-01"',
                    '"2015-05-01">=index>="2015-03-01"', 
                    '"2020-05-01">=index>="2020-03-01"'
                    ]
time_name =     [   'Taper tantrum',
                    'Fed shrink the balance sheet',
                    'China', 
                    'Covid-19'
                ]
time_exper_ctr = [  '"2013-07-01">=index>="2013-05-01"',
                    '"2018-03-01">=index>="2018-01-01"',
                    '"2015-07-01">=index>="2015-04-01"', 
                    '"2020-04-01">=index>="2020-03-13"'
                    ]                    
#%%
covid_ctr =port_return_df_risk.copy()# port_return_df_risk.copy()
g=0
for time_exper in time_exper_ctr:
    g+=1
    change_loc = covid_ctr.query(time_exper).index
    covid_ctr.loc[change_loc] =port_return_df_risk_ctr[['benchmark' for i in range(9)]] #port_return_df_risk_ctr.loc[change_loc]
    # if g==4:
    #     covid_ctr.loc[change_loc] = port_return_df_risk_ctr.loc[change_loc]

#%%

for time_exper in time_exper_li:
    dr_table.query(time_exper).eval('factor_std/p_risk').plot(figsize=(12, 6))
    plt.show()

# %%
ii=0
colors = ['red', 'green']
ucol = ['benchmark', 'Hight_Exposure_porfolio']
for time_exper, name  in zip(time_exper_ctr, time_name):#= '"2013-11-01">=index>="2013-06-01"' #'"2020-05-01">=index>="2020-03-16"'
    print(name)
    ii+=1
    for factor in style_factor+['risk_factor']:
        if ii!=5:

            ax = (port_return_df_risk_ctr[['benchmark', factor]]
                .assign(Hight_Exposure_porfolio= port_return_df_risk[factor])
                #.rename(columns={factor:'Low_Exposure_porfolio'})
                .loc[:, ucol]
                .rename(columns={'benchmark':'Dynamic portfolio', "Hight_Exposure_porfolio":'Aggressive portfolio'})
                .query(time_exper).cumsum()#.query('"2020-05-01">index>"2020-01-01"')
                #.eval('benchmark-Hight_Exposure_porfolio')
                #.eval(factor+'-risk')
                .mul(100)
                .plot(title=factor, figsize=(12, 6),color=dict(zip(colors, ucol)))
                )
        else:
            ax = (port_return_df_risk_ctr[['benchmark', factor]]
                .assign(Hight_Exposure_porfolio= port_return_df_risk[factor])
                .rename(columns={factor:'Low_Exposure_porfolio'})
                .query(time_exper).cumsum()#.query('"2020-05-01">index>"2020-01-01"')
                #.eval('benchmark-Hight_Exposure_porfolio')
                #.eval('Low_Exposure_porfolio - Hight_Exposure_porfolio')
                .mul(100)
                .plot(title=factor, figsize=(12, 6))
                )
        if factor=='risk_factor':
            ax.set_title('Multi-factor',fontsize=16)
        else:
            ax.set_title(factor,fontsize=16)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        plt.savefig('../pic/timing/{time}/{f}.jpg'.format(f=factor, time=name),bbox_inches='tight', pad_inches=0)
        plt.show()


# %%

all_time = "{s}>index>={e}".format(s="'2021-01-01'", e="'2011-03-01'")
in_sample = "{s}>index>={e}".format(s="'2016-01-01'", e="'2011-03-01'")
out_sample = "{s}>index>={e}".format(s="'2021-01-01'", e="'2016-01-01'")
test_exper = all_time

trac_table =  (port_return_df_risk
                    .drop('benchmark', axis=1)
                    .apply(lambda s:s - port_return_df_risk.benchmark)
                    .query(test_exper)
)
track_error = (trac_table
                    .std()*np.sqrt(252)
                    )

track_alpha = (trac_table
                    .pipe(lambda df: (df.sum()*252)/len(df))
                    )

track_ic = track_alpha/track_error
track_ic
# %%
ctr_table = (port_return_df_risk_ctr
                    .drop('benchmark', axis=1)
                    .apply(lambda s:s-port_return_df_risk.benchmark)
                    .query(test_exper)
                    )

track_error_ctr = (ctr_table
                    .std()*np.sqrt(252)
                    )

track_alpha_ctr = (ctr_table
                    .pipe(lambda df: 252*(df.sum()/len(df)))
                    )

track_ic_ctr = track_alpha_ctr/track_error_ctr
track_ic_ctr
# %%
dina_table = (covid_ctr
                    .drop('benchmark', axis=1)
                    .apply(lambda s:s-port_return_df_risk.benchmark)
                    .query(test_exper)
)
track_error_dina = (dina_table
                    .std()*np.sqrt(252)
                    )

track_alpha_dina = (dina_table
                    .pipe(lambda df: 252*(df.sum()/len(df)))          
                    )

track_ic_ctr_dina = track_alpha_dina/track_error_dina
track_ic_ctr_dina

# %%
IC_table =  pd.concat([track_ic, track_ic_ctr,track_ic_ctr_dina], axis=1).T
IC_table.index=['Low retain ratio', 'High retain ratio',
    'Dynamic retain ratio'
    ]

#IC_table.to_csv('../result/IC_table.csv')
IC_table
#%%

alpha_table =  pd.DataFrame()
alpha_table['High_TD'] = track_alpha_ctr
alpha_table['High_TE'] = track_error_ctr
alpha_table['Low_TD'] = track_alpha
alpha_table['Low_Te'] = track_error

#alpha_table.T.to_csv('../result/TD_TE_table.csv')


#%%

port_return_df_risk.query(test_exper).to_csv("Enhance_return.csv")
port_return_df_risk_ctr.query(test_exper).to_csv("Enhance_return_ctr.csv")
covid_ctr.query(test_exper).to_csv("Enhance_return_ctr_new.csv")
# %%
for col in port_return_df_risk.columns[1: ]:
    if col=='Muti-factor':
        continue
    pdf = (port_return_df_risk.loc[:, ['benchmark', col]]
            .query(test_exper)
            .cumsum().mul(100)
            .rename(columns={'benchmark': 'LQD'})
            )
    if col=='risk_factor':
        pdf = pdf.rename(columns={'risk_factor':'Muti-factor'})
        
    axes = pdf.plot(figsize=(12, 6))
    if col=='risk_factor':
        axes.set_title('Performance Comparison Multi-factor',fontsize=16)
    else:
        axes.set_title('Performance Comparison '+col, fontsize=16)
    axes.yaxis.set_major_formatter(mtick.PercentFormatter())
    axes.set_xlabel('Date')
    axes.set_ylabel('Cumulative return')
    # plt.savefig('../pic/enhance/{f}.jpg'.format(f=col)
    #         ,bbox_inches='tight'
    #         ,pad_inches=0)
    plt.show()

#%%
annual_return = [i.query(test_exper).mean()*252 for i in [port_return_df_risk, port_return_df_risk_ctr, covid_ctr]]
pd.concat(annual_return, axis=1).T.to_csv('../result/return_ana.csv', index=False)
# %%
