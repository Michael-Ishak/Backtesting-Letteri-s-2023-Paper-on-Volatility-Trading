import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTES
# Strategy follows after Ivan's paper of AITA (Artificial Intelligence Technical Analysis)
# OHLC is Open, High, Low, Close (part of PA [price action])

# Translating the modules and equation:

#########################################
# VolTS Historical Volatility (HV) module
#########################################

def parkinson_estimator(high, low, number_of_observations):
    
    constant = 1/(4*number_of_observations*np.log(2))
    
    variable = 0

    for i in range(1, number_of_observations+1):
        variable += np.log((high[i]/low[i])**2)
    
    PK_squared = constant * variable
    return np.sqrt(PK_squared)

# Garman-Klass (GK) estimator

def garman_klass(high, low, close, open, number_of_observations):

    constant = 1/number_of_observations

    first_variable = 0
    second_variable = 0

    for i in range(1, number_of_observations + 1):
        first_variable += 0.5 * np.power((np.log(high[i]/low[i])),2)
        second_variable += (2*np.log(2) - 1) *( (np.log(close[i]/open[i]))**2 )
    
    garman_klass_squared = constant * (first_variable - second_variable)

    return np.sqrt(garman_klass_squared)

# Rogers-Satchell (RS) estimator

def rogers_satchell(high, low, close, open, number_of_observations):

    constant = 1/number_of_observations

    first_variable = 0
    second_variable = 0

    for i in range(1, number_of_observations + 1):
        first_variable += np.log(high[i]/close[i])*np.log(high[i]/open[i])
        second_variable += np.log(low[i]/close[i])*np.log(low[i]/open[i])
    
    rogers_satchell_squared = constant * (first_variable + second_variable)

    return np.sqrt(rogers_satchell_squared)

# Yang-Zhang (YZ) estimator

def open_to_close_volatility(close, open, number_of_observations):
    
    constant = 1/(number_of_observations - 1)

    variable = 0

    for i in range(1,number_of_observations + 1):
        variable += (np.log(close[i]/open[i]) - np.log(close[i]/open[i]))**2

    return (constant * variable)

# def overnight_volatility(close, open, number_of_observations):
   
#    constant = 1/(number_of_observations - 1)

#    variable = 0
   
#    for i in range(1, number_of_observations):
#         variable += (np.log(open[i]/close[i - 1]) - np.log(open[i]/close[i - 1]))**2

#     return (constant * variable)

# def yang-zhang(overnight_vol, open_to_close_vol, RS, number_of_observations):
