# Getting Message frequency signal from Time domain signal
#---------------------------------------------------------

# Prepare a time domain signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import pi

sample_rate = 100
N = (2 - 0) * sample_rate  # N size of array, 2 seconds data is available so 200 samples available

time = np.linspace(0, 2, N)
# np.linspace(start, end, no of samples to generate evenly spaced)
# np.arrange(begin_time, end_time, samplingInterval (time b/w 2 consecutive samples)

# Create time data
freq1 = 20
amplitude1 = 1
waveform1 = amplitude1 * np.sin (2 * pi * freq1 * time)
time_data = waveform1

# Plot time domain signal
plt.plot (time,time_data)
plt.title ('Time Domain Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()

# Function fft from scipy.fftpack (takes array as input & converts into freq domain)

# According to Nyquist-Shannon Sampling theorem,
# we can only analyze frequency components up to half of the sampling rate.

frequency = np.linspace (0.0, sample_rate/2, int (N/2))
print(frequency)

freq_data = fft(time_data)

# Function returns both +ve and -ve frequencies, we only want +ve frequencies
# so use absolute function of numpy
y = 2/N * np.abs (freq_data [0:np.int (N/2)])

# Frequency spectrum to plot. Frequencies on the x-axis and frequency data for y-axis.

plt.plot(frequency, y)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

#----------------
# Definitions
#----------------
# Message frequency - (no of cycles in 1 second)
# Sampling frequency - (no of samples seen in 1 second)
#
# Omega (2*π*Fm (in terms of freq) or 2*π/T)
# Sine Function  waveform1 = amplitude1 * np.sin (2 * pi * freq1 * time)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
# ax[0].plot(x, y)     # plot time series
# ax[1].stem(x_f, y_f) # plot freq domain
