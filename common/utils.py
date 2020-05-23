import numpy as np
from common import crc, hamming


def convert_dec2bin(decimal: int, size: int):
    tmp = bin(decimal)
    binary = []
    for i in range(2, len(tmp)):
        binary.append(int(tmp[i]))
    if len(binary) < size:
        binary = [0]*(size-len(binary)) + binary
    return binary


def convert_bin2dec(binary: list):
    decimal = ''
    for bit in binary:
        decimal += str(bit)
    return int(decimal, 2)


def encode_and_modulate(data_in, algorithm: str, k: int, key: list = None):
    bulk_codes = []

    def modulate_bpsk(bit: int):
        if bit:
            return 1
        else:
            return -1
    if algorithm == 'hamming':
        for j in data_in:
            code = hamming.encode(convert_dec2bin(j, k))
            modulated_code = np.array(list(map(modulate_bpsk, code)))

            bulk_codes.append(modulated_code)
    elif algorithm == 'cyclic':
        if key is None:
            print('Indica el polinomio generador')
            return
        for j in data_in:
            code = crc.encode(convert_dec2bin(j, k), key)
            modulated_code = list(map(modulate_bpsk, code))
            bulk_codes.append(np.array(modulated_code))
    bulk_codes = np.array(bulk_codes)
    return bulk_codes


def demodulate_and_decode(data_in, algorithm: str, dmin: int = 3, key: list = None):
    bulk_values = []

    if algorithm == 'hamming':
        for j in data_in:
            demodulated_code = list(map(lambda x: int(x>=0), j))
            predicted_bits = hamming.decode(demodulated_code)
            predicted_value = convert_bin2dec(predicted_bits)
            bulk_values.append(predicted_value)
    elif algorithm == 'cyclic':
        if key is None:
            print('Indica el polinomio generador')
            return
        for j in data_in:
            demodulated_code = list(map(lambda x: int(x>=0), j))
            predicted_bits = crc.decode(demodulated_code, key, dmin)
            predicted_value = convert_bin2dec(predicted_bits)
            bulk_values.append(predicted_value)

    return bulk_values


class Soft:
    def __init__(self, algorithm: str, k: int, dmin: int = None, key: list = None):
        self.algorithm = algorithm
        self.k = k
        self.dmin = dmin
        self.key = key

        self.codes = []
        for i in range(2 ** self.k):
            if self.algorithm == 'hamming':
                self.codes.append(hamming.encode(convert_dec2bin(i, self.k)))
            elif self.algorithm == 'cyclic':
                self.codes.append(crc.encode(convert_dec2bin(i, self.k), self.key))
        self.codes = np.array(self.codes)

    def soft_demodulate_and_decode(self, data_in):
        bulk_values = []

        def demodulate_bpsk(symbol):
            symbol = np.array(symbol)
            modulated_codes = np.vectorize(lambda x: x*2 - 1,)(self.codes)
            op = modulated_codes-symbol
            distances = np.multiply(op,op).sum(axis=1)
            return list(self.codes[distances.argmin()])

        if self.algorithm == 'hamming':
            for j in data_in:
                demodulated_code = demodulate_bpsk(j)
                predicted_bits = hamming.decode(demodulated_code)
                predicted_value = convert_bin2dec(predicted_bits)
                bulk_values.append(predicted_value)
        elif self.algorithm == 'cyclic':
            if self.key is None:
                print('Indica el polinomio generador')
                return
            for j in data_in:
                demodulated_code = demodulate_bpsk(j)
                predicted_bits = crc.decode(demodulated_code, self.key, self.dmin)
                predicted_value = convert_bin2dec(predicted_bits)
                bulk_values.append(predicted_value)

        return bulk_values
