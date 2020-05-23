import numpy as np
import matplotlib.pyplot as plt
from common import Filter
from scipy.signal import unit_impulse
from common.utils import encode_and_modulate, demodulate_and_decode, Soft

# defining parameters
k = 4
n_channel = 7
R = k / n_channel
M = 2**k
# key = [1,1,1,0,1,0,0,0,1]  # Polinomio generador
# dmin = 5
dmin = 3
print('M:', M, 'k:', k, 'n:', n_channel, 'Rate:', R)

N = 100000
test_label = np.random.randint(M, size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)

test_data = np.array(test_data)

EbNo = 10.0**0.1
noise_mean = 0
nn = N
t = np.arange(N*n_channel)
freq = np.fft.fftfreq(t.size, d=0.5)

noise = np.random.randn(N*n_channel)
sp_white = np.fft.fft(noise.flatten())/(N*n_channel)
noise = np.random.randn(nn, n_channel+ 2)
final_signal = Filter.noise_equalizer_1(noise)
sp_color = np.fft.fft(final_signal.flatten())/(N*n_channel)
plt.subplot(2, 1, 1)
plt.plot(freq, np.absolute(sp_white)**2)
plt.xlabel('Normalized to Pi frequency [rad/s]')
plt.ylabel('Spectral density [W/Hz]')
plt.grid()
plt.title('AWGN Noise')

plt.subplot(2, 1, 2)
plt.plot(freq, np.absolute(sp_color)**2)
plt.xlabel('Normalized to Pi frequency [rad/s]')
plt.ylabel('Spectral density [W/Hz]')
plt.grid()
plt.title('Equalized Noise')

plt.show()
