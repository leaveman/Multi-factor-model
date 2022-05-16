# -*- coding: utf-8 -*-
"""
Created on Sat Mar  29 11:59:07 2021

@author: LEAVEMAN
"""
import pandas as pd
import pyodbc
import numba as nb
import scipy.stats as stats
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.linalg import pinv
import urllib
import sqlalchemy
import matplotlib.pyplot as plt
import numexpr as ne
from numba import vectorize, njit
from joblib import Parallel, delayed
import multiprocessing
# --------------------------------------------------------


def sql_conect(sql_comander, db):
    server = '(local)'
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' +
                        server + ';DATABASE=' + db + ';Trusted_Connection=yes')
    sql = sql_comander
    df = pd.io.sql.read_sql(sql, conn)
    return df


def get_sql_engine():
    server = '(local)'
    db = 'leaveman'
    params = urllib.parse.quote_plus(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')
    engine = sqlalchemy.create_engine(
        "mssql+pyodbc:///?odbc_connect=%s" % params)
    return engine
# --------------------------------------------------------


def strip_table(df):
    temp_df = df.copy()
    not_na_loc = df['code'].str.strip().notna().copy()
    temp_df['code'].loc[not_na_loc] = df['code'].str.strip().loc[not_na_loc]
    return temp_df
# --------------------------------------------------------


# 切換對角線
def change_diag(a, b):
    return b-np.diag(b.diagonal())+np.diag(a.diagonal())


# 指數移動平均係數
@vectorize
def exp_weight(x, halflife):
    return (0.5**(1/halflife))**x


# 單一矩陣正定檢定
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# --------------------------------------------------------
def split_time(df, sp_day=False):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    if sp_day:
        df['day'] = df['date'].dt.days


def add_date(df):
    pre = df
    pre['sdate_m'] = pre['year'].map(str) + '-' + pre['month'].map(str)
    pre['sdate_m'] = pd.to_datetime(pre['sdate_m'], format='%Y-%m')
    return pre


def tej_to_date(df):
    new_df = df.copy()
    new_df['date'] = pd.to_datetime(new_df['date'], format='%Y%m%d')
    return new_df


def market_preprocess(s):
    cap = np.sqrt(s)
    max_95 = np.percentile(cap, 95)
    cap = np.where(cap > max_95, max_95, cap)
    return cap




# ------------------------------------------
# 台灣財報日期對應函數
# ------------------------------------------

def fr_date_transform(date):
    y = date.year
    m = date.month
    d = date.day

    if y < 2013:
        if 4 > m:
            return str(y-1)+'-3'
        elif 5 > m >= 4:
            return str(y-1)+'-4'
        elif 9 > m >= 5:
            return str(y)+'-1'
        elif 11 > m >= 9:
            return str(y)+'-2'
        elif 12 >= m >= 11:
            return str(y)+'-3'

    else:
        if 4 > m:
            return str(y-1)+'-3'
        elif 5 > m >= 4:
            return str(y-1)+'-4'
        elif 8 > m >= 5:
            if d <= 15:
                return str(y-1)+'-4'
            else:
                return str(y)+'-1'
        elif 11 > m >= 8:
            if d <= 14:
                return str(y)+'-1'
            else:
                return str(y)+'-2'
        elif 12 >= m >= 11:
            if d <= 14:
                return str(y)+'-2'
            else:
                return str(y)+'-3'
# --------------------------------------------------------


def standarlize_w(array, weight):
    w_sum = sum(weight)
    #result = ne.evaluate('(A-mean(A*W, 0)/sum(w))/std',{'A':array, 'W':weight,'std':np.std(array, axis=0)})
    result =(array - (np.mean(array*weight, axis=0)/w_sum)) / np.std(array, axis=0)
    return result


def outliear(array):
    temp_array = array
    ind_o = np.where(array > 3)
    ind_u = np.where(array < -3)

    sp = np.maximum(0, np.minimum(1, 0.5 / (np.max(array, axis=0)-3)))
    sn = np.maximum(0, np.minimum(1, 0.5 / (-np.min(array, axis=0)-3)))

    splus = np.take(sp, ind_o[1])
    snegi = np.take(sn, ind_u[1])

    temp_array[ind_o] = 3*(1-splus)+array[ind_o]*splus
    temp_array[ind_u] = 3*(1-snegi)+array[ind_u]*snegi

    return temp_array
# ------------------------------------------
# Dataframe分割類
# ------------------------------------------


def df_split(df, group_key, stype='list'):

    if group_key == 'index':
        if stype == 'list':
            return [table for _, table in df.groupby(level=0)]
        elif stype == 'dict':
            return {keys: table for keys, table in df.groupby(level=0)}
        else:
            print('No such stype')

    elif group_key in df.columns:
        if stype == 'list':
            return [table for _, table in df.groupby(group_key)]
        elif stype == 'dict':
            return {keys: table for keys, table in df.groupby(group_key)}
        else:
            print('No such stype')
    else:
        print('group_key Not in columns')
# --------------------------------------------------
# Barra eue3 Two steps regression
# ------------------------------------------


class Barra_Reg:
    def __init__(self, df, factor, industry):

        self.df = df  # dataframe
        self.sfa = factor  # list of style factor
        self.ind = industry  # list of industry factor
        self.all_factor = factor + industry  # list of all factor
        self.factor_num = len(self.all_factor)

        self.x_data = df[self.all_factor].values
        self.y_data = df['return'].values[: ,np.newaxis]

        self.weight = (df['Mvalue']
                        .pipe(np.sqrt)
                        .pipe(lambda s: s/sum(s))
                        .to_numpy()
                        )

    # 計算產業權重

    def ind_weight(self):
        i_weight = (self.df
                    .groupby('industry').Mvalue
                    .sum()
                    .pipe(lambda s: s/np.sum(s))
                    .to_numpy()
                    )

        return i_weight

    def Omega(self):
        '''
        Barra 因子報酬矩陣
        '''

        X = self.x_data
        R_col_len = self.factor_num - 1
        ind_wiegt = self.ind_weight()        # 產業權重限制式
        ind_num = len(self.ind)


        R = np.eye(self.factor_num, R_col_len)
        R[R_col_len, 1: ind_num] = (-1)*ind_wiegt[: -1]/ind_wiegt[-1]
        V = np.diag(self.weight)

        inv_part = pinv(R.T @ X.T @ V @ X @ R)
        Ω = R @ inv_part @ R.T @ X.T@ V
        return Ω

    def factor_return(self):

        Ω = self.Omega()

        factor_return_first = Ω @ self.y_data
        residual = self.x_data @ factor_return_first - self.y_data

        sigu = np.median(abs(residual - np.median(residual)))
        jump = np.where(abs(residual) > (4*sigu),
                        np.sign(residual)*(abs(residual)-(4*sigu)),
                        0)

        second_y = self.y_data - jump
        factor_return = (Ω @ second_y).reshape(self.factor_num, )


        S = pd.Series(factor_return, index=self.all_factor)
        return S

    def residual(self):

        Ω = self.Omega()
        factor_return_first = Ω @ self.y_data
        residual = self.x_data @ factor_return_first - self.y_data
        residual = pd.Series(residual.ravel())
        return residual

    
    #R-sqared
    def r_squared(self):
        r_squared = 1 - ( self.weight.T@(self.residual().values**2) )/( self.weight.T@(self.y_data**2))
        return r_squared

# --------------------------------------------------


# %%
# --------------------------------------------------
# Barra EWMA covariance matrix
# --------------------------------------------------

#@nb.jit(nopython=True)
def roilling_exp_cov(array, exp_w, exp_sum, dstr):
    cov_len, rolling_len, fa_lan = array.shape
    
    array_t = array.transpose(0, 2, 1)*exp_w

    temp_array = (array_t@array).copy()   
    for i in range(1, dstr+1):
        newest_coef = 1-(i)/(dstr+1)
        end = rolling_len - i
        temp_array += Nw_with_ne(array_t[:,: , :end], array[:, i: ,: ], newest_coef)

    return ne.evaluate("22*temp_array/exp_sum")

def roilling_exp_var(array, exp_w, exp_sum, dstr):
    cov_len, rolling_len, fa_lan = array.shape
    
    array_t = array.transpose(0, 2, 1) * np.sqrt(exp_w)

    temp_array = ne.evaluate('sum(array_t**2, axis=2)')  
    for i in range(1, dstr+1):
        newest_coef = 2*(1-i/(dstr+1))
        end =  -i
        temp_array += Nw_with_ne(array_t[:,: , :end], array_t[:, :, i: ] , newest_coef, False)
    return ne.evaluate("22*temp_array/exp_sum")


def Nw_with_ne(f, fp, b, is_cov=True):
    if is_cov:
        dstr_ma = (f@fp)
        return b*(dstr_ma + dstr_ma.transpose(0,2,1))
    else:
        var_ma = b*ne.evaluate('sum(f*fp, 2)')
        return var_ma

@njit
def Barra_sp_risk(s, window, half_life , dstr):
    window_range = np.arange(window-1, -1, -1)
    exp_w_var = exp_weight(window_range, half_life)
    exp_w_var_sum = np.sum(exp_w_var)
    var = np.sum(exp_w_var*(s**2))/exp_w_var_sum
    return var

@nb.jit(nopython=True)
def newest_adj_f(array, dstr, exp_w, exp_sum, rolling_len, i):
    newest_adj = 0
    for j in range(1, dstr+1):
        newest_coef = 1-(j)/(dstr+1)
        end = rolling_len - j
        Gamma_0_c = array[i, 0:end, :].T.copy()
        Gamma_j = array[i, j:rolling_len]
        dstr_ma = np.dot(Gamma_0_c*exp_w[:, 0:end], Gamma_j)/exp_sum
        dstr_ma += dstr_ma.T
        newest_adj = newest_adj + dstr_ma*newest_coef
        return newest_adj


class Barra_moving_cov:

    # 初始參數
    def __init__(self, df, window, half_life, dstr):
        self.df = df
        self.w = window
        self.hf = half_life
        self.dstr = dstr

    # 獲取處理後index

    def cov_index(self):
        ind_date = self.df.index[self.w: ]
        ind_fa = self.df.columns
        ind = pd.MultiIndex.from_product([ind_date, ind_fa], names=['Date', 'Factor'])
        return ind

    # Rolling取出存放進3D array

    def rolling_array(self):
        v = self.df.values
        d0, d1 = v.shape
        s0, s1 = v.strides
        window_size = self.w + 1
        rolling_array = as_strided(v, 
                                (d0 - (window_size - 1), window_size, d1),
                                (s0, s0, s1)
                                ).copy()
        r_s0, _, r_s2 = rolling_array.shape
        rolling_array -= np.mean(rolling_array, axis=2)[:, :, np.newaxis]
        return rolling_array

    # 計算當期 EWMA COV 與New-west 調整

    def moving_cov(self):
        rol_array = self.rolling_array()
        d0, rolling_len, d1 = rol_array.shape

        window_range = np.arange(rolling_len-1, -1, -1).reshape(1, rolling_len)

        exp_w = exp_weight(window_range, self.hf)
        exp_sum = np.sum(exp_w)

        exp_w_var = exp_weight(window_range, self.hf/2)
        exp_var_sum = np.sum(exp_w_var)

        rol_array_corr = roilling_exp_cov(rol_array,
                                        exp_w,
                                        exp_sum,
                                        self.dstr)

        rol_array_var = roilling_exp_var(rol_array,
                                        exp_w_var,
                                        exp_var_sum, 
                                        self.dstr)
        idx = np.arange(d1)

        rol_array_corr[:, idx, idx] = rol_array_var
        rol_array_corr = rol_array_corr.reshape(d0*d1, d1)
        temp_df = pd.DataFrame(rol_array_corr, columns=self.df.columns, index=self.cov_index())

        return temp_df

    # 將Array轉換成Dataframe
    def moving_var(self):
        rol_array = self.rolling_array()
        d0, rolling_len, d1 = rol_array.shape

        window_range = np.arange(rolling_len-1, -1, -1).reshape(1, rolling_len)
        exp_w_var = exp_weight(window_range, self.hf)
        exp_var_sum = np.sum(exp_w_var)

        rol_array_var = roilling_exp_var(rol_array,
                                            exp_w_var,
                                            exp_var_sum,
                                            self.dstr)

        temp_df = pd.DataFrame(rol_array_var,
                                columns=self.df.columns,
                                index=self.df.tail(d0).index)
        return temp_df
# --------------------------------------------------

