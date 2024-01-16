import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.neighbors import NearestNeighbors

# NOTES
# Strategy follows after Letteri's paper of AITA (Artificial Intelligence Technical Analysis)
# OHLC is Open, High, Low, Close (part of PA [price action])

# Translating the modules and equation:

###########################################
# VolTS Historical Volatility (HV) module #
###########################################


def parkinson_estimator(high, low, number_of_observations):
    constant = 1 / (4 * number_of_observations * np.log(2))

    variable = 0

    for i in range(1, number_of_observations + 1):
        variable += np.log((high[i] / low[i]) ** 2)

    PK_squared = constant * variable
    return np.sqrt(PK_squared)


# Garman-Klass (GK) estimator


def garman_klass(high, low, close, open, number_of_observations):
    constant = 1 / number_of_observations

    first_variable = 0
    second_variable = 0

    for i in range(1, number_of_observations + 1):
        first_variable += 0.5 * np.power((np.log(high[i] / low[i])), 2)
        second_variable += (2 * np.log(2) - 1) * ((np.log(close[i] / open[i])) ** 2)

    garman_klass_squared = constant * (first_variable - second_variable)

    return np.sqrt(garman_klass_squared)


# Rogers-Satchell (RS) estimator


def rogers_satchell(high, low, close, open, number_of_observations):
    constant = 1 / number_of_observations

    first_variable = 0
    second_variable = 0

    for i in range(1, number_of_observations + 1):
        first_variable += np.log(high[i] / close[i]) * np.log(high[i] / open[i])
        second_variable += np.log(low[i] / close[i]) * np.log(low[i] / open[i])

    rogers_satchell_squared = constant * (first_variable + second_variable)

    return rogers_satchell_squared, np.sqrt(rogers_satchell_squared)


# Yang-Zhang (YZ) estimator


def yang_zhang(close, open, number_of_observations, sigma_RS_squared):
    constant = 1 / (number_of_observations - 1)

    k = 10.34 / (1.34 + (number_of_observations + 1) / (number_of_observations - 1))

    sigma_open_squared = 0
    overnight_average = 0
    open_to_close_average = 0
    sigma_overnight_squared = 0

    # Computing averages

    for i in range(1, number_of_observations + 1):
        overnight_average += np.log(open[i] / close[i - 1]) / number_of_observations
        open_to_close_average += np.log(close[i] / open[i]) / number_of_observations

    # Computing sigmas

    for j in range(1, number_of_observations + 1):
        sigma_open_squared += (np.log(close[j] / open[j]) - open_to_close_average) ** 2
        sigma_overnight_squared += (
            np.log(open[j] / [close[j - 1]]) - overnight_average
        ) ** 2

    yang_zhang_squared = (
        (constant) * (sigma_overnight_squared)
        + (constant * k) * sigma_open_squared
        + (1 - k) * (sigma_RS_squared)
    )

    return np.sqrt(yang_zhang_squared)


##########################
# AITA Strategies module #
##########################


class AITA:
    def __init__(self, close, time, volume, period):
        self.close = close
        self.volume = volume
        self.time = time
        self.period = period

    def Buy_and_Hold(self, close, time, volume):
        initial_price = 0
        if time[0]:
            initial_price = close[0]
        V_0 = initial_price * volume
        V_252 = (volume * close[-1]) - V_0

        return V_252

    def Trend_Following(self, close, period, weighting_factor):
        ema = np.zeros(len(close))
        sma = np.mean(close[:period])

        ema[period - 1] = sma

        for i in range(period, len(close)):
            ema[i] = (close[i] * weighting_factor) + (
                ema[i - 1] * (1 - weighting_factor)
            )

        boolean = 0

        if close[0] > ema[0]:
            # Uptrend
            boolean = 1

        if close[0] < ema[0]:
            # Downtrend
            boolean = 2
        return boolean

    def Mean_Reversion(self, close, k, open_long, open_short):
        # K is the number of standard deviations from the mean
        mu = np.mean(close)
        sigma = np.std(close)

        input_trade_operation = False

        if close[0] < (mu - (k * sigma)):
            if open_long != 0:
                if open_short > 0:
                    # CLOSE SHORT CLOSE SHORT
                    pass
                else:
                    # GO LONG GO LONG GO LONG
                    input_trade_operation = True
        elif close[0] > (mu - (k * sigma)):
            if open_short != 0:
                if open_long > 0:
                    # CLOSE LONG CLOSE LONG
                    pass
                else:
                    # GO SHORT GO SHORT GO SHORT
                    input_trade_operation = True
        return input_trade_operation


#############################
# AitaBT Backtesting module #
#############################


class AitBT:
    def __init__(self, close, risk_free_rate, number_of_observations, time_step):
        self.close = close
        self.risk_free_rate = risk_free_rate
        self.number_of_observations = number_of_observations
        self.time_step = time_step

    def max_drawdown(self, close):
        returns = close.pct_change()
        cumulative_returns = (1 + returns).cumprod()
        max_cumulative = cumulative_returns.cummax()
        drawdown = (cumulative_returns - max_cumulative) / max_cumulative

        return "%.2f" % (drawdown.min() * 100) + "%"

    def sharpe_ratio(self, close, risk_free_rate, number_of_observations):
        mu = np.mean(close) * number_of_observations - risk_free_rate
        sigma = np.std(close) * np.sqrt(number_of_observations)

        return mu / sigma

    def sortino_ratio(self, close, risk_free_rate, number_of_observations):
        mu = np.mean(close) * number_of_observations - risk_free_rate
        negative_sigma = np.std(close[close < 0]) * np.sqrt(number_of_observations)

        return mu / negative_sigma

    def calmar_ratio(self, close, number_of_observations):
        return (
            np.mean(close) * number_of_observations / (np.abs(self.max_drawdown(close)))
        )

    def calculate_returns_multiple_symbols(close, time_step):
        p, n = close.shape

        total_returns = np.zeros((p, n))

        for k in range(p):
            for t in range(1, n):
                total_returns[k, t] = (close[k, t + time_step] - close[k, t]) / close[
                    k, t
                ]

        mu = np.mean(total_returns, axis=1)
        sigma = np.std(total_returns, axis=1)

        standardised_returns = (total_returns - mu[:, np.newaxis]) / sigma[
            :, np.newaxis
        ]

        return total_returns, standardised_returns


##############################################################
# Importing the Daily OHLC for the Assets Named in the Paper #
##############################################################

OHLC_df = yf.download(
    ["MSFT", "GOOGL", "MU", "NVDA", "AMZN", "META", "QCOM", "IBM", "INTC"],
    start="2014-01-01",
    end="2024-01-01",
    auto_adjust=True,
)

print(OHLC_df)

##########################
# AITA Anomaly Detection #
##########################

# Fit a KNN model to the time series data - Ensure adj_close is used to adjust for dividends, splits etc.
# For each point, calculate distance to neighbors and anomaly score
# Determine threshold based on mean and standard deviation of scores
# Find points with score greater than threshold as anomalies