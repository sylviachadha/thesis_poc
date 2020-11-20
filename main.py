# Synthetic sine wave generation
# Aim - To estimate the pattern / cycle length using fft
##--------------------------------------------------------

## Method 1 #################################

## Reference
# https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
# https://pythontic.com/visualization/signals/fouriertransform_fft
# https://medium.com/@khairulomar/deconstructing-time-series-using-fourier-transform-e52dd535a44e
# https://www.youtube.com/watch?v=c249W6uc7ho&feature=youtube
# https://www.youtube.com/watch?v=su9YSmwZmPg

# Synthetic Data - Amplitude 1, Frequency 10Hz, Cycles 20
#-----------------------------------------------------------
# Step1 -  Import Libraries
#------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from numpy.fft import fft, fftfreq, ifft
import pandas as pd

# Step2 - Generate Sine Function
#--------------------------------
Fm = 100 # frequency (no of cycles in 1 second) T=1/100 * Also denote by Lx 1 cycle time
Fs = 1000 # sampling freq (no of samples seen in 1 second i.e in 1000 points seen in 100 cycles here)

T = 1/Fm # Distance in meters or time period in seconds)
print(Lx) # Time period for 1 cycle


n=200 # total points u need
T=T
# Creating data in terms of samples
x_1 = np.linspace(0, T, n)


n=1000
T = 100 # points in 1 cycle 1/Fm
x=np.linspace(0,T,n)
# Total 1000 points and 1 cycle need to cover in 100 points so he wants to show data for 10 cycles
# He conveys how much total data he needs in terms of total no of points
print(x)

# Another method specify in terms of time in seconds (10 sec of data) instead of samples (1000)
x=(0,10,)




# np.arange(begin_time, end_time, samplingInterval)
x = np.arange(0,2,1/Fs)  # 1/Fs will give sampling interval- time diff b/w 2 consecutive samples

n=200 #Total sample points = Fs * Lx)
#x_1 = np.linspace(0, Lx, n)
# Choose either x or x_1 both mean same

# angular frequency = omega which is 2*π*Fm (in terms of freq) or 2*π/T (in terms of Time Period)
omega = 2*np.pi/Lx
#omega1 = 2*np.pi*Fm
# Both omega and omega1 same because Fm=1/Lx

# Sine function
# 10 denoting wave is going to occur 10 times in that particular domain
y = 1*np.sin(10*omega*x)
y = np.sin(2 * np.pi * Fm * x)
# Plot for the synthetic sine wave
plt.plot(x,y)
plt.show()

# Step 3 - Change t to t_f (time to frequency domain)
#------------------------------------------------------

# Using Nyquist frequency Fs/2 to change t to t_f (time to frequency domain)
n = np.size(x)
print(n)

# Method 1 to calculate x_f
x_f = Fs/2 * np.linspace(0,2,n//2)  # np.linspace(start, stop, no of samples to be generated in b/w)
print(x_f)
# Fn = Fs/2 which is Nyquist frequency
# Sampling freq reduced by half Fs/2 so now u only check 50 times in 1 cycle so samples also half n/2


# Method 2
values = np.arange(int(len(y) / 2))
timePeriod = len(y) / Fs
x_f = values / timePeriod

# Method 3



# Step 4 - Change y to y_f (time signal to frequency signal)
#-------------------------------------------------------------

# Perform Fourier transform using scipy
#y_fft = fftpack.fft(y)
y_f = fftpack.fft(y)
#y_f = 2/n * abs(y_fft[0:np.size(x_f)])

# Since u changed x 200 to x_f 100 so u needed to change in y_f to 100 also ??????????????????????????


# Step 5 - Plot Data
#-----------------------

# Plot 1 - Time and Frequency plots

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot(x, y)     # plot time series
ax[1].stem(x_f, y_f) # plot freq domain
plt.show()

# Plot 2 - Another way to show frequency plot

fig, ax = plt.subplots()
ax.plot(x_f, y_f)
plt.show()








