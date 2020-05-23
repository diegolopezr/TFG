import numpy as np


def noise_equalizer_1(input):
    a = 0.7
    b = 0.5     # w[n] = 1 + a*z^-1 + b*z^-2
    n = input[0].shape[0] - 2
    w_n = np.zeros((n+2, n))
    for i in range(n):    # w[n] = 1 + a*z^-1 + b*z^-2
        w_n[i][i] = b
        w_n[i+1][i] = a
        w_n[i+2][i] = 1
    noise_filter_energy = (np.array([b, a, 1], dtype=np.float32) ** 2).sum()
    result = np.matmul(input, w_n) / np.sqrt(noise_filter_energy)
    return result

def noise_equalizer_2(input):
    a = 0.7
    b = 0     # w[n] = 1 + a*z^-1
    n = input[0].shape[0] - 2
    w_n = np.zeros((n+2, n))
    for i in range(n):    # w[n] = 1 + a*z^-1
        w_n[i][i] = b
        w_n[i+1][i] = a
        w_n[i+2][i] = 1
    noise_filter_energy = (np.array([b, a, 1], dtype=np.float32) ** 2).sum()
    result = np.matmul(input, w_n) / np.sqrt(noise_filter_energy)
    return result


def signal_filter_2(input):
    n = 7
    filter = np.zeros((2 * n, n))
    FIR_coeficients = np.array([0.1224, -0.1635, -0.0159, 0.3492, -0.4571, -0.0584, 0.9961], dtype=np.float32)
    for i in range(n):
        filter[i + 1][i] = FIR_coeficients[0]
        filter[i + 2][i] = FIR_coeficients[1]
        filter[i + 3][i] = FIR_coeficients[2]
        filter[i + 4][i] = FIR_coeficients[3]
        filter[i + 5][i] = FIR_coeficients[4]
        filter[i + 6][i] = FIR_coeficients[5]
        filter[i + 7][i] = FIR_coeficients[6]
    input_delayed = np.concatenate((np.zeros((1,n)),input[:-1]))   # Adelantados una posicion
    concatenated = np.concatenate((input_delayed, input), axis=1)
    signal_filter_energy = (np.array([0.1224, -0.1635, -0.0159, 0.3492, -0.4571, -0.0584, 0.9961], dtype=np.float32) ** 2).sum()
    result = np.matmul(concatenated, filter) / np.sqrt(signal_filter_energy)
    return result


def signal_filter_3(input):
    n = 7
    filter = np.zeros((2 * n, n))
    FIR_coeficients = np.array([-0.0799, 0.1262, -0.0168, -0.2289, 0.3540, -0.0379, -0.655, 0.9929], dtype=np.float32)
    for i in range(n):
        filter[i][i] = FIR_coeficients[0]
        filter[i + 1][i] = FIR_coeficients[1]
        filter[i + 2][i] = FIR_coeficients[2]
        filter[i + 3][i] = FIR_coeficients[3]
        filter[i + 4][i] = FIR_coeficients[4]
        filter[i + 5][i] = FIR_coeficients[5]
        filter[i + 6][i] = FIR_coeficients[6]
        filter[i + 7][i] = FIR_coeficients[7]
    input_delayed = np.concatenate((np.zeros((1,n)),input[:-1]))   # Adelantados una posicion
    concatenated = np.concatenate((input_delayed, input), axis=1)
    signal_filter_energy = (np.array([-0.0799, 0.1262, -0.0168, -0.2289, 0.3540, -0.0379, -0.655, 0.9929], dtype=np.float32) ** 2).sum()
    result = np.matmul(concatenated, filter) / np.sqrt(signal_filter_energy)
    return result
