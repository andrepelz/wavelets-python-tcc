import numpy as np
from numpy.typing import ArrayLike

SIGNAL_NAME = 'speech'
INPUT_DATA_FILENAME = f'{SIGNAL_NAME}.pcm'
NOISY_DATA_FILENAME = f'noisy_{SIGNAL_NAME}.pcm'
NOISE_FILENAME = f'{SIGNAL_NAME}_noise.pcm'


def get_input_data_from_file() -> ArrayLike:
    filename = INPUT_DATA_FILENAME
    with open(filename, 'rb') as input_file:
        signal = np.fromfile(input_file, dtype=np.int16)

    return signal


def get_noisy_data_from_file() -> ArrayLike:
    filename = NOISY_DATA_FILENAME
    with open(filename, 'rb') as input_file:
        noisy = np.fromfile(input_file, dtype=np.int16)

    return noisy


input_data = get_input_data_from_file()
noisy_data = get_noisy_data_from_file()

noise = noisy_data - input_data[:noisy_data.size]

filename = NOISE_FILENAME
with open(filename, 'wb') as output_file:
    output_file.write(noise)

filename = INPUT_DATA_FILENAME
with open(filename, 'wb') as output_file:
    output_file.write(input_data[:noisy_data.size])
