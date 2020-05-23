import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import gaussian, equalized_1, equalized_2, equalized_3
from common import Filter

# defining parameters
k = 4
n_channel = 7
R = k / n_channel
M = 2**k
dmin = 3
print('M:', M, 'k:', k, 'n:', n_channel, 'Rate:', R)

# _____________________________________________________________________________________________________________________

encoder_g, decoder_g = gaussian.create_autoencoder(n_channel, k)
encoder1, decoder1 = equalized_1.create_autoencoder(n_channel, k)
encoder2, decoder2 = equalized_2.create_autoencoder(n_channel, k)
encoder3, decoder3 = equalized_3.create_autoencoder(n_channel, k)

N = 10000000
test_label = np.random.randint(M, size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)

test_data = np.array(test_data)

# _____________________________________________________________________________________________________________________

def frange(x, y, jump):
     while x < y:
         yield x
         x += jump


EbNodB_range = list(frange(-4, 8.5, 0.5))
autoencoder_g_bler = [None] * len(EbNodB_range)
autoencoder_1_bler = [None] * len(EbNodB_range)
autoencoder_2_bler = [None] * len(EbNodB_range)
autoencoder_3_bler = [None] * len(EbNodB_range)
for n in range(len(EbNodB_range)):
    EbNo = 10.0**(EbNodB_range[n]/10.0)
    noise_mean = 0
    nn = N

    # Autoencoder Gaussian BLER
    encoded_signal = encoder_g.predict(test_data)
    noise_std = np.sqrt(1 / (2 * k * EbNo))
    final_signal = encoded_signal + noise_std * np.random.randn(nn, n_channel)
    pred_final_signal = decoder_g.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    autoencoder_g_bler[n] = no_errors / nn

    # Autoencoder Eq 1 BLER
    encoded_signal = encoder1.predict(test_data)
    noise_std = np.sqrt(1 / (2 * k * EbNo))
    final_signal = encoded_signal + Filter.noise_equalizer_1(noise_std * np.random.randn(nn, n_channel+2))
    pred_final_signal = decoder1.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    autoencoder_1_bler[n] = no_errors / nn

    # Autoencoder Eq 2 BLER
    encoded_signal = encoder2.predict(test_data)
    noise_std = np.sqrt(1 / (2 * k * EbNo))
    final_signal = Filter.signal_filter_2(encoded_signal) + Filter.noise_equalizer_2(noise_std * np.random.randn(nn, n_channel + 2))
    pred_final_signal = decoder2.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    autoencoder_2_bler[n] = no_errors / nn

    # Autoencoder Eq 3 BLER
    encoded_signal = encoder3.predict(test_data)
    noise_std = np.sqrt(1 / (2 * k * EbNo))
    final_signal = Filter.signal_filter_3(encoded_signal) + noise_std * np.random.randn(nn, n_channel)
    pred_final_signal = decoder3.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    autoencoder_3_bler[n] = no_errors / nn


plt.plot(EbNodB_range, autoencoder_g_bler, 'bo', label=f'Autoencoder({n_channel},{k}) AWGN')
plt.plot(EbNodB_range, autoencoder_1_bler, 'go', label=f'Autoencoder({n_channel},{k}) ZF Eq.')
plt.plot(EbNodB_range, autoencoder_2_bler, 'ro', label=f'Autoencoder({n_channel},{k}) ZF Incomplete Eq.')
plt.plot(EbNodB_range, autoencoder_3_bler, 'ko', label=f'Autoencoder({n_channel},{k}) No Eq.')

# _____________________________________________________________________________________________________________________

plt.yscale('log')
plt.xlabel('SNR Range [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='lower left', ncol=1)

plt.show()
