# importing libs
import numpy as np
import tensorflow as tf
from scipy.signal import unit_impulse
from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.optimizers import Adam
from keras import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization


def train(model, training_data):
    model.fit(training_data, training_data,
              epochs=60,
              batch_size=900,
              verbose=2,
              validation_split=0.1)


def add_impulse_noise(x, a=-10, b=10, c=0.0001):
    """
    PDF for impulse noise is f(x) = c / ( b - a ) + ( 1 - c ) * d( x ), where d(x) is the delta Dirach
    :param a:  Min value for impluse noise
    :param b:  Max value for impluse noise
    :param c:  Probability that impulse noise happens
    :return:
    """
    def _pdf(x_, a, b, c):	# Returns a vector with same length of x_ containing the PDF probabilities
        discrete_probabilities_list = []
        for i in x_:
            discrete_probabilities_list.append(c/(b-a))
        discrete_probabilities_list = discrete_probabilities_list + (1-c)*unit_impulse(len(discrete_probabilities_list),'mid')
        return discrete_probabilities_list

    values = np.arange(10*a,10*b+1)/10
    pdf = _pdf(values,a,b,c)
    pdf = pdf/pdf.sum()  # Assert that the sum of all discrete probabilities equals 1
    shape=x.shape
    result = x + np.random.choice(values, size=shape, p=pdf)
    return result


def create_autoencoder(n, k, c):
    M = 2 ** k
    R = k / n
    # generating data of size N
    N = 500000
    label = np.random.randint(M, size=N)

    # creating one hot encoded vectors
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)

    data = np.array(data)

    input_signal = Input(shape=(M,), name='Signal_input')
    encoded = Dense(M, activation='relu', name='relu')(input_signal)
    encoded1 = Dense(n, activation='linear', name='linear')(encoded)
    encoded2 = BatchNormalization()(encoded1)
    encoded2_ = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='Energy_normalizing')(encoded2)

    EbNo_train = 10 ** 2.7  # converted 7 dB of EbNo
    encoded3 = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded2_)
    encoded4 = Lambda(lambda x: x + add_impulse_noise(np.zeros(1), a=-10*np.sqrt(1 / (2 * R * EbNo_train)), b=10*np.sqrt(1 / (2 * R * EbNo_train)), c=c), name='Add_impulsive_noise')(encoded3)

    decoded = Dense(M, activation='relu', name='relu_2')(encoded4)
    decoded1 = Dense(M, activation='softmax', name='softmax')(decoded)

    autoencoder = Model(input_signal, decoded1)
    adam_opti = Adam(lr=0.0008)
    autoencoder.compile(optimizer=adam_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    print(autoencoder.summary())

    train(autoencoder, data)

    encoder = Model(input_signal, encoded2_)  # Model representing the encoder part

    encoded_input = Input(shape=(n,))  # New input code sized

    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)  # Previous trained model with calculated weights representing the decoder

    return encoder, decoder
