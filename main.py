import numpy as np
import pandas as pd
import pywt

from numpy.typing import ArrayLike
from scipy.io import wavfile


GLOBAL_MAX_LEVEL = 5
SIGNAL_SAMPLE_RATE = 8000

SIGNAL_NAME = 'speech-librivox-0062'
SIGNAL_FOLDER = 'musan/speech/librivox'
NOISE_NAME = 'noise-free-sound-0287'
NOISE_FOLDER = 'musan/noise/free-sound'
OUTPUT_FOLDER = f'outputs/{SIGNAL_NAME}'

FILE_EXTENSION = 'wav'

INPUT_DATA_FILENAME = f'{SIGNAL_FOLDER}/{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISE_FILENAME = f'{NOISE_FOLDER}/{NOISE_NAME}.{FILE_EXTENSION}'

INPUT_DATA_COPY_FILENAME = f'{OUTPUT_FOLDER}/{SIGNAL_NAME}.{FILE_EXTENSION}'
OUTPUT_DATA_FILENAME = f'{OUTPUT_FOLDER}/denoised_{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISY_DATA_FILENAME = f'{OUTPUT_FOLDER}/noisy_{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISE_COPY_FILENAME = f'{OUTPUT_FOLDER}/noise_{NOISE_NAME}.{FILE_EXTENSION}'
REMAINING_NOISE_FILENAME = f'{OUTPUT_FOLDER}/remaining_noise.{FILE_EXTENSION}'
RESULTS_FILENAME = f'{OUTPUT_FOLDER}/results.csv'


signal_sample_rate = 0
noise_sample_rate = 0


def init_mother_wavelets() -> list[str]:
    result = []

    daubechies = [ 'db4', 'db8', 'db16' ]
    result = np.append(result, daubechies)

    symlets = [ 'sym4', 'sym8' ]
    result = np.append(result, symlets)

    return result


def init_threshold_types() -> list[str]:
    return [ 'hard', 'soft' ]

def init_k_coeffs() -> list[float]:
    return [ 0.25, 0.5, 0.75, 1.0 ]

def init_m_coeffs() -> list[float]:
    return [ 0.2, 0.4, 0.6, 0.8, 1.0 ]


def get_input_data_from_file() -> ArrayLike:
    global signal_sample_rate

    filename = INPUT_DATA_FILENAME

    if FILE_EXTENSION == 'pcm':
        with open(filename, 'rb') as input_file:
            signal = np.fromfile(input_file, dtype=np.int16)
    elif FILE_EXTENSION == 'wav':
        signal_sample_rate, signal = wavfile.read(filename)
    else:
        raise ValueError('Unknown/Unsupported file extension.')

    return signal


def get_noise_from_file() -> ArrayLike:
    global noise_sample_rate

    filename = NOISE_FILENAME

    if FILE_EXTENSION == 'pcm':
        with open(filename, 'rb') as input_file:
            noise = np.fromfile(input_file, dtype=np.int16)
    elif FILE_EXTENSION == 'wav':
        noise_sample_rate, noise = wavfile.read(filename)
    else:
        raise ValueError('Unknown/Unsupported file extension.')

    return noise


def save_outputs_to_file(input_data: ArrayLike, noise: ArrayLike, noisy_data: ArrayLike, output_data: ArrayLike, remaining_noise: ArrayLike) -> None:
    from pathlib import Path
    import os
    
    if len(input_data.shape) > 1:
        input_data = np.transpose(input_data)
        noise = np.transpose(noise)[0]
        noisy_data = np.transpose(noisy_data)
        output_data = np.transpose(output_data)
        remaining_noise = np.transpose(remaining_noise)

    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, f'{OUTPUT_FOLDER}')

    Path(full_path).mkdir(parents=True, exist_ok=True)

    if FILE_EXTENSION == 'pcm':
        filename = INPUT_DATA_COPY_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(input_data)

        filename = NOISE_COPY_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(noise)

        filename = NOISY_DATA_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(noisy_data)

        filename = OUTPUT_DATA_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(output_data)

        filename = REMAINING_NOISE_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(remaining_noise)
    elif FILE_EXTENSION == 'wav':
        filename = INPUT_DATA_COPY_FILENAME
        wavfile.write(filename, signal_sample_rate, input_data)

        filename = NOISE_COPY_FILENAME
        wavfile.write(filename, noise_sample_rate, noise)

        filename = NOISY_DATA_FILENAME
        wavfile.write(filename, signal_sample_rate, noisy_data)

        filename = OUTPUT_DATA_FILENAME
        wavfile.write(filename, signal_sample_rate, output_data)

        filename = REMAINING_NOISE_FILENAME
        wavfile.write(filename, signal_sample_rate, remaining_noise)
    else:
        raise ValueError('Unknown/Unsupported file extension.')


def normalize(data: ArrayLike) -> ArrayLike:
    return data/np.power(2, 15)


def unnormalize(data: ArrayLike) -> ArrayLike:
    return (data*np.power(2, 15)).astype(np.int16)


def calculate_threshold(d1: ArrayLike, size: int, k: float = 0.8, m: float = 0.8) -> float:
    return k*m*np.median(np.abs(d1))/0.6745*np.sqrt(2*np.log(size))


def snr(signal: ArrayLike, noise: ArrayLike) -> float:
    signal_power = np.mean(np.square(signal.astype(np.float64), dtype=np.float64))
    noise_power = np.mean(np.square(noise.astype(np.float64), dtype=np.float64))

    return 10*np.log10(signal_power/noise_power)


def mse(original_signal: ArrayLike, resulting_signal: ArrayLike) -> float:
    return np.mean(np.square(original_signal - resulting_signal))


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

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=local_max_level)

    wavelet_coefficients = transform[1:]
    coefficients_d1 = wavelet_coefficients[-1] # coefficients D1 from first level wavelet decomposition
    threshold = calculate_threshold(coefficients_d1, input_data.size, k_coeff, m_coeff) # threshold calculated for coefficients D1

    for index, coefficients in enumerate(wavelet_coefficients):
        wavelet_coefficients[index] = pywt.threshold(coefficients, value=threshold, mode=threshold_type)
    
    transform[1:] = wavelet_coefficients

    output_data = pywt.waverec(transform, mother_wavelet)
    output_data = output_data[:input_data.size].astype(np.int16)

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
        'output_mse': output_mse
    }

    save_outputs_to_file(input_data, noise, noisy_data, output_data, remaining_noise)

    return values


def main():
    mother_wavelets = init_mother_wavelets()
    print(f'{mother_wavelets=}')

    global_max_level = GLOBAL_MAX_LEVEL
    print(f'{global_max_level=}')

    threshold_types = init_threshold_types()
    print(f'{threshold_types=}')

    k_coeffs = init_k_coeffs()
    print(f'{k_coeffs=}')

    m_coeffs = init_m_coeffs()
    print(f'{m_coeffs=}')

    data = get_input_data_from_file()
    noise = get_noise_from_file()

    print(f'{data=}')
    print(f'{noise=}')

    if data.size > noise.size:
        data = data[:noise.size]
    else:
        noise = noise[:data.size]

    if len(data.shape) > 1: # adjust noise for stereo audio
        data = np.transpose(data)
        noise = np.array([noise, noise])
        
    noise = noise//2

    results = []

    current_run = 1
    total_runs = (
        len(mother_wavelets)
        *(global_max_level - 1)
        *len(threshold_types)
        *len(k_coeffs)
        *len(m_coeffs)
    )

    for wavelet in mother_wavelets:
        for level in range(1, global_max_level):
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
                        
    # results.append(
    #     evaluate_noise_reduction_algorithm(
    #         data, 
    #         noise, 
    #         mother_wavelets[0], 
    #         global_max_level, 
    #         threshold_types[0], 
    #         k_coeffs[0], 
    #         m_coeffs[0]))

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(RESULTS_FILENAME)


if __name__ == '__main__':
    main()
