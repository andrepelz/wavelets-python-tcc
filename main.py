import numpy as np
import pandas as pd
import pywt
from time import perf_counter_ns
from research.modwt import modwt_waverec, modwt_wavedec

from variables import *
from utils import generate_white_noise, mse, snr, normalize, calculate_threshold
from io_handling import get_input_data_from_file, get_noise_from_file, save_results

from numpy.typing import ArrayLike

# SIGNAL_SAMPLE_RATE = 16000

SIGNAL_NAME = 'speech-librivox-0117'
NOISE_NAME = 'noise-free-sound-0165'
INPUT_FOLDER = f'inputs/masculina'
NOISE_FOLDER = f'inputs/ruido/real'
OUTPUT_FOLDER = f'outputs/{NOISE_NAME}/{SIGNAL_NAME}&{NOISE_NAME}'

FILE_EXTENSION = 'wav'

INPUT_DATA_FILENAME = f'{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISE_FILENAME = f'{NOISE_NAME}.{FILE_EXTENSION}'


def evaluate_noise_reduction_algorithm(
    input_data: ArrayLike, 
    noise: ArrayLike,
    mother_wavelet: str, 
    max_level: int, 
    threshold_type: str,
    k_coeff: float,
    m_coeff: float
) -> dict:
    local_max_level = min(max_level, pywt.dwt_max_level(input_data.size, mother_wavelet))

    if local_max_level < max_level: # if local max level is reduced, then further calculations will be duplicated redundant
        return (None, None)
    
    noisy_data = input_data + noise

    input_mse = mse(normalize(input_data), normalize(noisy_data))
    input_snr = snr(input_data, noise)

    execution_start = perf_counter_ns()

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=local_max_level)

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
        'local_max_level': local_max_level,
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
    mother_wavelets = init_mother_wavelets()
    transform_max_level = init_transform_level()
    threshold_types = init_threshold_types()
    k_coeffs = init_k_coeffs()
    m_coeffs = init_m_coeffs()

    signal_sample_rate, data = get_input_data_from_file(INPUT_DATA_FILENAME, INPUT_FOLDER, FILE_EXTENSION)
    noise_sample_rate, noise = get_noise_from_file(NOISE_FILENAME, NOISE_FOLDER, FILE_EXTENSION)
    noise = noise//3

    data = data[:signal_sample_rate*30]

    # noise = generate_white_noise(data, 10)[:signal_sample_rate*30]
    # global noise_sample_rate
    # noise_sample_rate = signal_sample_rate

    if data.size > noise.size:
        data = data[:noise.size]
    else:
        noise = noise[:data.size]

    if len(data.shape) > 1: # adjust noise for stereo audio
        data = np.transpose(data)
        noise = np.array([noise, noise])
        
    results = []

    current_run = 1
    total_runs = (
        len(mother_wavelets)
        *(transform_max_level)
        *len(threshold_types)
        *len(k_coeffs)
        *len(m_coeffs)
    )

    for wavelet in mother_wavelets:
        for level in range(1, transform_max_level + 1):
            for thresh_type in threshold_types:
                for k_coeff in k_coeffs:
                    for m_coeff in m_coeffs:
                        print(f'Running transform {current_run}/{total_runs}')
                        current_run += 1
                        results.append(
                            evaluate_noise_reduction_algorithm(
                                data, 
                                noise, 
                                wavelet, 
                                level, 
                                thresh_type, 
                                k_coeff, 
                                m_coeff
                            )
                        )

    print()

    results = pd.DataFrame(results)
    save_results(results, data, noise, INPUT_DATA_FILENAME, NOISE_FILENAME, OUTPUT_FOLDER, FILE_EXTENSION, signal_sample_rate)


if __name__ == '__main__':
    main()
