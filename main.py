import numpy as np
import pandas as pd
import pywt
from time import perf_counter_ns

from variables import *
from utils import update_noise_to_target_snr, mse, snr, normalize, calculate_threshold
from io_handling import get_input_data_from_file, get_noise_from_file, save_results, _save_artifacts_from_configuration

from numpy.typing import ArrayLike

import sys

# SIGNAL_SAMPLE_RATE = 16000

INPUT_NAME = 'speech-librivox-0005'
NOISE_NAME = 'white_noise'
INPUT_FOLDER = f'inputs/portugues'
NOISE_FOLDER = f'inputs/ruido/branco'
OUTPUT_FOLDER = f'outputs/{NOISE_NAME}/{INPUT_NAME}&{NOISE_NAME}'

FILE_EXTENSION = 'wav'

INPUT_DATA_FILENAME = f'{INPUT_NAME}.{FILE_EXTENSION}'
NOISE_FILENAME = f'{NOISE_NAME}.{FILE_EXTENSION}'

TARGET_INPUT_SNR = 5


def evaluate_noise_reduction_algorithm(
    input_data: ArrayLike, 
    noise: ArrayLike,
    mother_wavelet: str, 
    max_level: int, 
    threshold_type: str,
    k_coeff: float,
    m_coeff: float
) -> dict:
    noisy_data = input_data + noise

    input_mse = mse(normalize(input_data), normalize(noisy_data))
    input_snr = snr(input_data, noise)

    execution_start = perf_counter_ns()

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=max_level)

    wavelet_coefficients = transform[1:]
    coefficients_d1 = wavelet_coefficients[-1] # coefficients D1 from first level wavelet decomposition
    threshold = calculate_threshold(coefficients_d1, input_data.size, k_coeff, m_coeff) # threshold calculated for coefficients D1

    for index, coefficients in enumerate(wavelet_coefficients):
        wavelet_coefficients[index] = pywt.threshold(coefficients, value=threshold, mode=threshold_type)
    
    transform[1:] = wavelet_coefficients

    output_data = pywt.waverec(transform, mother_wavelet)
    output_data = output_data[:input_data.size].astype(np.int16)

    execution_end = perf_counter_ns()

    remaining_noise = output_data - input_data

    output_mse = mse(normalize(input_data), normalize(output_data))
    output_snr = snr(input_data, remaining_noise)

    values = { 
        'mother_wavelet': mother_wavelet,
        'local_max_level': max_level,
        'threshold_type': threshold_type,
        'k_coeff': k_coeff,
        'm_coeff': m_coeff,
        'input_snr': input_snr, 
        'output_snr': output_snr, 
        'input_mse': input_mse, 
        'output_mse': output_mse,
        'execution_time': (execution_end - execution_start)/1000000 # in milliseconds
    }

    return values


def main():
    mother_wavelet = sys.argv[3]
    level = int(sys.argv[4])
    threshold_type = sys.argv[5]
    k_coeff = float(sys.argv[6])
    m_coeff = float(sys.argv[7])
    target_snr = float(sys.argv[8])

    if len(sys.argv) != 9:
        return -1

    input_name = sys.argv[1]
    noise_name = sys.argv[2]
    input_folder = f'test/{input_name}'
    noise_folder = f'test/{input_name}'
    output_folder = f'test/{input_name}'
    file_extension = 'wav'
    input_filename = f'{input_name}.{file_extension}'
    noise_filename = f'{noise_name}.{file_extension}'

    signal_sample_rate, data = get_input_data_from_file(input_filename, input_folder, file_extension)
    _, noise = get_noise_from_file(noise_filename, noise_folder, file_extension)

    if data.size > noise.size:
        data = data[:noise.size]
    else:
        noise = noise[:data.size]

    noise = update_noise_to_target_snr(noise, data, target_snr)

    _save_artifacts_from_configuration(
        data, 
        noise, 
        mother_wavelet, 
        level, 
        threshold_type, 
        k_coeff, 
        m_coeff,
        input_filename,
        noise_filename,
        output_folder,
        file_extension,
        signal_sample_rate
    )


if __name__ == '__main__':
    main()
