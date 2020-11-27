# Correlogram / Autocorrelation Plot / ACF and PACF Function
# Correlation of series with itself lagged by x time units
# 1 time lag - how correlated is today's sale to previous day sale

# x-axis - no of time units of lag
# y-axis - correlation values
import heapq
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import pandas as pd


# One Cycle / Pattern Extraction
# --------------------------------
##############################################################
# Example 1 Synthetic Sine wave
# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

# Load dataset

df = pd.read_csv('sine_synthetic.csv', index_col='t')
df.describe()
df.head()
print(len(df))

# Expectation
# Since freq is 10Hz so 10 cycles in 1 second, so T=1/10 i.e 0.1 to
# cover 1 cycle.
# Fs = 100 so 100 samples seen in 1 sec that is in 10 cycles so
# 10 samples in 1 cycle (so lag should be on previous 10 values for 1 full cycle)

df['y'].plot()
plt.show()

values = acf(df['y'])
correlation_values = np.round(values, 2)
print(correlation_values)
type(correlation_values)  # its an array
#correlation_values1 = [10,280,40,110] # Example

#y_list = df['y'].to_list()
#y_list

# gcd for correlation values list
# def findgcd(x, y):
#     while (y):
#         x, y = y, x % y
#     return x
#
# l = correlation_values
# num1 = l[0]
# num2 = l[1]
# gcd = findgcd(num1, num2)
# for i in range(2, len(l)):
#     gcd = findgcd(gcd, l[i])
# print("gcd is: ", gcd)

# Largest values in array
heapq.nlargest(5, correlation_values)
# Largest values along with index to get period
# https://medium.com/@yurybelousov/the-beauty-of-python-or-how-to-get-indices-of-n-maximum-values-in-the-array-d362385794ef
max_correlated = heapq.nlargest(5, range(len(correlation_values)), correlation_values.take)
max_correlated_index = sorted(max_correlated)
print(max_correlated_index)

# Plot ACF Function
plot_acf(df['y'], lags=60)
plt.show()

#########################################################################
# Example 2 Synthetic Triangle wave
# Load dataset

df = pd.read_csv('Triangular_wave_1.csv', index_col='time')
df.describe()
df.head()
print(len(df))

# Plot Series

ax = df.plot(figsize=(12, 6), title='Triangular Wave',
             marker='o', markerfacecolor='black', markersize=5);
xlabel = 'time'
ylabel = 'amplitude'
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

# Plot ACF Function
plot_acf(df['amplitude'], lags=30)
plt.show()

# Correlation values
values = acf(df['amplitude'])
correlation_values = np.round(values, 2)
print(correlation_values)
type(correlation_values)

# Largest values in array
heapq.nlargest(5, correlation_values)
# Check Period
max_correlated = heapq.nlargest(5, range(len(correlation_values)), correlation_values.take)
max_correlated_index = sorted(max_correlated)
print(max_correlated_index)


# Definitions-
# Correlation - measure of the strength of linear relationship b/w 2 variables
# Correlation Values from -1 to +1
# near 1 -> stronger +ve linear relationship (both go up or dowm)
# near -1 -> stronger -ve linear relationship (one goes up, other goes down)
# near 0 -> weaker linear relationship or no association

# Example 3 ##############################################################
# Real ECG Dataset

os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

df = pd.read_csv('ecg108_test.csv', index_col='t')
df.describe()
df.head()
print(len(df)) # 60 seconds or 1 min data
# In general 60 cardiac cycle data in 1 min (if 60 beats/min)

# We need to find out whether recorded pattern is
# 1 sec cycle or 0.75 sec cycle or 0.60 or 1.5 sec cycle??

# 1 sec data is covered in 360 samples (Fs=0.002 sec)
# Total data we have is 21600 samples available (360*60)

# Cycle Duration

# First anomaly at 4105 sample i.e. in 11 cycles
# 10 seconds or 3600 samples only to check pattern
# (estimated 10 cycles to be seen in 10 sec data)

# Full 60 seconds data plot
df['y'].plot()
plt.show()

# Give slicer in terms of time duration (x axis is time)
newdf = df.drop(['Label', 'x'], axis=1)

newdf.head()
df_filter = newdf[:11]
print(len(df_filter))
df_filter.plot()
plt.show()


## To decide no. of lags and how many largest values - 3000 lags
values = acf(df_filter['y'],nlags=1500)
correlation_values = np.round(values, 2)
print(len(correlation_values))
type(correlation_values)

plot_acf(df_filter['y'], lags=1500)
plt.show()

# Largest values in array
largest_n_values = heapq.nlargest(80, correlation_values)
print(largest_n_values)
# Check Period
max_correlated = heapq.nlargest(150, range(len(correlation_values)), correlation_values.take)
#max_correlated_index = sorted(max_correlated)
print(max_correlated)  ## Appx 350 points

# Plot 2 cycles of 350 points each
# Given data 360 samples  ->> 1 sec
# 350 samples means recorded cycle of 0.9 sec



