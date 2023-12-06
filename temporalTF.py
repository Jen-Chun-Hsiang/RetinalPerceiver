import numpy as np
import matplotlib.pyplot as plt

# Define the sampling rate and time vector T
SamplingRate = 40
T = np.arange(-1, 3, 1/SamplingRate)
result_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Figures/'
cfid = 1
# Define a Gaussian function
def gaussian(x, sigma, mean):
    return np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))

# Define the function Ypf with T as input
def Ypf(T, w):
    gauss1 = gaussian(T, w[0], w[2]) * w[4]
    gauss2 = gaussian(T, w[1], w[3]) * w[5]
    return (gauss1 - gauss2) + w[6]

# Calculate freqf_t
T_positive = T[(T >= 0) & (T < 0.5)]
freqf_t = Ypf(T_positive, [0.05, 0.12, 0.08, 0.12, 1, 0.2, 0])

# Plotting freqf_t
plt.figure(figsize=(10, 6))
plt.plot(T_positive, freqf_t, label='freqf_t')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Frequency Function freqf_t')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'{result_dir}check_figure{cfid}.png')