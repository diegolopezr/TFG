import numpy as np


def encode(data: list):
    data = np.array(list(map(int, data)))
    generator_matrix = np.array([[1,0,0,0,1,0,1],[0,1,0,0,1,1,1],[0,0,1,0,1,1,0],[0,0,0,1,0,1,1]])
    code = np.matmul(data, generator_matrix)
    return list(map(lambda x: np.mod(x,2),code))


def decode(data):
    data = np.array(list(map(int, data)))
    h_t = np.array([[1,0,1],[1,1,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1]])
    syndrome = np.matmul(data, h_t)
    syndrome = list(map(lambda x: np.mod(x,2),syndrome))
    if 1 not in syndrome:
        return data[:4]
    error_patterns = []
    for i in range(4):
        error = [0]*7
        error[i] = 1
        err_syndrome = list(np.matmul(error, h_t))
        error_patterns.append(err_syndrome)
    for i in range(4):
        if syndrome == error_patterns[i]:
            data[i] = data[i] ^ 1
            return data[:4]
    return data[:4]

