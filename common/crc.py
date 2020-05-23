import numpy as np


def reminder(divident, divisor):
    pick = len(divisor)
    tmp = divident[:pick]

    while pick < len(divident):
        if tmp[0] == 1:
            for i in range(1, len(tmp)):
                tmp[i - 1] = tmp[i] ^ divisor[i]
            tmp[-1] = divident[pick]
        else:
            for i in range(1, len(tmp)):
                tmp[i - 1] = tmp[i] ^ 0
            tmp[-1] = divident[pick]
        pick += 1

    if tmp[0] == 1:
        for i in range(1, len(tmp)):
            tmp[i - 1] = tmp[i] ^ divisor[i]
    else:
        for i in range(1, len(tmp)):
            tmp[i - 1] = tmp[i] ^ 0

    tmp.pop(-1)
    return tmp


def cyclic_shift(vector, shifts: int, length: int):
    full_vector = [0]*(length-len(vector)) + vector
    for i in range(shifts):
        full_vector = full_vector[1:] + [full_vector[0]]
    return full_vector


def weight(vector: list):
    weight_count = 0
    for i in range(len(vector)):
        weight_count += vector[i]
    return weight_count


def encode(data: list, key: list):
    r = len(key) - 1
    k = len(data)
    data = list(map(int, data))
    key = list(map(int, key))

    padded_data = data + [0]*r
    b = reminder(padded_data, key)  # Calculate the Checksum
    return data + b


def decode(received_data: list, key: list, dmin: int):
    received_data = list(map(int, received_data))
    key = list(map(int, key))
    n = len(received_data)
    r = len(key)-1
    t = int((dmin-1)/2)  # Capacidad correctora del codigo
    syndrome = reminder(received_data, key)
    if 1 not in syndrome:
        return received_data[:-r]   # No ha habido errores
    i = 0
    error = None
    while i < n:
        if weight(syndrome) <= t:       # Error trapping method
            error = cyclic_shift(syndrome, n-i, n)
            break
        i += 1
        if i == n:
            return received_data[:-r]
        syndrome = syndrome[1:]+[0] if syndrome[0] == 0 else list(np.bitwise_xor(syndrome+[0], key))[1:]
    if error is not None:
        received_data = np.bitwise_xor(error, received_data)
        received_data = received_data.tolist()

    return received_data[:-r]
