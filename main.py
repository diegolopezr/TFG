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
from common.utils import encode_and_modulate, demodulate_and_decode, Soft
from scipy.stats import norm
from scipy.special import binom

# defining parameters
k = 4
n_channel = 7
R = k / n_channel
M = 2**k
dmin = 3
print('M:', M, 'k:', k, 'n:', n_channel, 'Rate:', R)

# _____________________________________________________________________________________________________________________

# encoder, decoder = gaussian.create_autoencoder(n_channel, k)
encoder, decoder = equalized_1.create_autoencoder(n_channel, k)

N = 1000000
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
autoencoder_bler = [None] * len(EbNodB_range)
soft_hamming_bler = [None] * len(EbNodB_range)
soft_cyclic_bler = [None] * len(EbNodB_range)
for n in range(len(EbNodB_range)):
    EbNo = 10.0**(EbNodB_range[n]/10.0)
    noise_mean = 0
    nn = N

    # Autoencoder BLER
    encoded_signal = encoder.predict(test_data)
    noise_std = np.sqrt(1 / (2 * k * EbNo))
    #final_signal = encoded_signal + noise_std * np.random.randn(nn, n_channel)
    final_signal = encoded_signal + Filter.noise_equalizer_1(noise_std * np.random.randn(nn, n_channel+2))
    pred_final_signal = decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    autoencoder_bler[n] = no_errors / nn
    print('SNR:', f'{EbNodB_range[n]} dB', 'Autoencoder BLER:', autoencoder_bler[n])

    noise_std = np.sqrt(1/(2*R*EbNo))
    noise = noise_std * np.random.randn(nn, n_channel+2)

    # Soft Hamming BLER
    sf = Soft('hamming', k)
    encoded_signal = encode_and_modulate(test_label, 'hamming', k)
    final_signal = encoded_signal + Filter.noise_equalizer_1(noise)
    pred_output = sf.soft_demodulate_and_decode(final_signal)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    soft_hamming_bler[n] = no_errors / nn
    print('SNR:', f'{EbNodB_range[n]} dB', 'Soft Hamming BLER:', soft_hamming_bler[n])

    # Soft CRC BLER
    # sf = Soft('cyclic', k, dmin, key)
    # encoded_signal = encode_and_modulate(test_label, 'cyclic', k, key)
    # final_signal = add_impulse_noise(encoded_signal, a*noise_std, b*noise_std, c) + noise
    # pred_output = sf.soft_demodulate_and_decode(final_signal)
    # no_errors = (pred_output != test_label)
    # no_errors = no_errors.astype(int).sum()
    # soft_cyclic_bler[n] = no_errors / nn
    # print('SNR:', f'{EbNodB_range[n]} dB', 'Soft Cyclic BLER:', soft_cyclic_bler[n])

    print('<\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\>')

# _____________________________________________________________________________________________________________________

EbNodB_range2 = list(frange(-4, 8.5, 0.5))
theoretical_bpsk_ber = [0] * len(EbNodB_range2)
experimental_bpsk_ber = [0] * len(EbNodB_range2)
theoretical_ber = [0] * len(EbNodB_range2)
hamming_bler = [None] * len(EbNodB_range2)
cyclic_bler = [None] * len(EbNodB_range2)
for n in range(len(EbNodB_range2)):
    EbNo = 10.0**(EbNodB_range2[n]/10.0)
    noise_mean = 0
    nn = N
    noise_std = np.sqrt(1/(2*R*EbNo))
    noise = noise_std * np.random.randn(nn, n_channel+2)

    # BPSK theoretical BER
    # theoretical_bpsk_ber[n] = norm.sf(np.sqrt(EbNo))

    # Hamming theoretical BLER
    # p = norm.sf(np.sqrt(2*EbNo*R))
    # for m in range(2, n_channel+1):
    #     theoretical_ber[n] += (binom(n_channel, m)*(p**m)*((1-p)**(n_channel-m)))

    # Hard Hamming BLER
    encoded_signal = encode_and_modulate(test_label, 'hamming', k)
    final_signal = encoded_signal + Filter.noise_equalizer_1(noise)
    pred_output = demodulate_and_decode(final_signal, 'hamming')
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    hamming_bler[n] = no_errors / nn
    print('SNR:', f'{EbNodB_range2[n]} dB', 'Hamming BLER:', hamming_bler[n])

    # Hard CRC BLER
    # encoded_signal = encode_and_modulate(test_label, 'cyclic', k, key)
    # final_signal = add_impulse_noise(encoded_signal, a * noise_std, b * noise_std, c) + noise
    # pred_output = demodulate_and_decode(final_signal, 'cyclic', dmin, key)
    # no_errors = (pred_output != test_label)
    # no_errors = no_errors.astype(int).sum()
    # cyclic_bler[n] = no_errors / nn
    # print('SNR:', f'{EbNodB_range2[n]} dB', 'Cyclic BLER:', cyclic_bler[n])


# plt.plot(EbNodB_range2, theoretical_bpsk_ber, 'k-', label='Theory Uncoded BPSK')
# plt.plot(EbNodB_range2, theoretical_ber, 'm-', label=f'Upper bound BlockCode({n_channel},{k})')
plt.plot(EbNodB_range2, hamming_bler, 'r-.', label=f'Hamming({n_channel},{k}) Hard Decision')
# plt.plot(EbNodB_range2, cyclic_bler, 'y-.', label=f'Cyclic({n_channel},{k}) Hard Decision')
plt.plot(EbNodB_range, autoencoder_bler, 'bo', label=f'Autoencoder({n_channel},{k})')
plt.plot(EbNodB_range, soft_hamming_bler, 'c--', label=f'Hamming({n_channel},{k}) Soft Decision')
# plt.plot(EbNodB_range, soft_cyclic_bler, 'g--', label=f'Cyclic({n_channel},{k}) Soft Decision')

# _____________________________________________________________________________________________________________________

plt.yscale('log')
plt.xlabel('SNR Range [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='lower left', ncol=1)

plt.show()
