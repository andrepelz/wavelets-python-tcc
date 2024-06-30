import numpy as np
import pandas as pd
import pywt
from time import perf_counter_ns
from research.modwt import modwt_waverec, modwt_wavedec

from variables import *
from utils import update_noise_to_target_snr, mse, snr, normalize, calculate_threshold
from io_handling import get_input_data_from_file, get_noise_from_file, save_results

from numpy.typing import ArrayLike

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
    transform_level: int, 
    threshold_type: str,
    k_coeff: float,
    m_coeff: float
) -> dict:
    """
    Função de avaliação do algoritmo de redução de ruídos usando Transformada Wavelet.

    Parâmetros
    ----------
    `input_data` : `ArrayLike`
        vetor com sinal de voz limpa de entrada
    `noise` : `ArrayLike`
        vetor com sinal de ruído
    `mother_wavelet` : `str`
        nome da Wavelet Mãe
    `transform_level` : `int`
        nível da transformada
    `threshold_type` : `str`
        nome do tipo de threshold
    `k_coeff` : `float`
        coeficiente k
    `m_coeff` : `float`
        coeficiente m
        
    Retorna
    -------
    `dict`
    """
    noisy_data = input_data + noise

    input_mse = mse(normalize(input_data), normalize(noisy_data))
    input_snr = snr(input_data, noise)

    execution_start = perf_counter_ns()

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=transform_level)

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
        'local_max_level': transform_level,
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
    """Função principal do programa."""
    mother_wavelets = init_mother_wavelets()
    transform_max_level = init_transform_level()
    threshold_types = init_threshold_types()
    k_coeffs = init_k_coeffs()
    m_coeffs = init_m_coeffs()

    signal_sample_rate, data = get_input_data_from_file(INPUT_DATA_FILENAME, INPUT_FOLDER, FILE_EXTENSION)
    _, noise = get_noise_from_file(NOISE_FILENAME, NOISE_FOLDER, FILE_EXTENSION)

    data = data[:signal_sample_rate*30]

    if data.size > noise.size:
        data = data[:noise.size]
    else:
        noise = noise[:data.size]

    noise = update_noise_to_target_snr(noise, data, TARGET_INPUT_SNR)

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
