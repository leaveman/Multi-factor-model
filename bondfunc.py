#%%
import numpy as np
import numba as nb
from  scipy import optimize
#%%
def bond_price(par, T, ytm, coup, freq=2):
        freq = float(freq)
        periods = int(T*freq)
        coupon = coup/100.*par/freq
        dt = (np.arange(periods)+1) / freq

        price = np.sum(coupon/(1+ytm/freq)**(freq*dt)) +  par/(1+ytm/freq)**(freq*T)
        return price


def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
        freq = float(freq)
        periods = int(T*freq)
        coupon = coup/freq
        dt = (np.arange(periods)+1) / freq
        ytm_func = lambda y:  np.sum(coupon/(1+y/freq)**(freq*dt)) +  par/(1+y/freq)**(freq*T) - price
        try:
                return optimize.newton(ytm_func, guess,maxiter=500)
        except:
                return np.nan


def bond_convexity(price, T, coup, freq, ytm, par=100, dy=0.01):
        if price == np.nan:
                return np.nan

        ytm_minus = ytm - dy
        price_minus = bond_price(par, T, ytm_minus, coup, freq)

        ytm_plus = ytm + dy
        price_plus = bond_price(par, T, ytm_plus, coup, freq)

        convexity = (price_minus+price_plus-2*price)/(price*dy**2)
        return convexity

v_bond_convexity = np.vectorize(bond_convexity) 
v_ytm_func = np.vectorize(bond_ytm) 
# %%

# %%

# %%
