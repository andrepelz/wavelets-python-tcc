import numpy as np
import pandas as pd
import pywt
from time import perf_counter_ns
from research.modwt import modwt_waverec, modwt_wavedec

from numpy.typing import ArrayLike
from scipy.io import wavfile


GLOBAL_MAX_LEVEL = 5
SIGNAL_SAMPLE_RATE = 16000

SIGNAL_NAME = 'speech-librivox-0005'
# SIGNAL_FOLDER = 'musan/speech/librivox'
NOISE_NAME = 'white_noise_5db'
# NOISE_FOLDER = 'musan/noise/free-sound'
INPUT_SUBFOLDER = 'portugues'
NOISE_SUBFOLDER = 'ruido'
INPUT_FOLDER = f'inputs'
OUTPUT_FOLDER = f'outputs/{SIGNAL_NAME}&{NOISE_NAME}'

FILE_EXTENSION = 'wav'

INPUT_DATA_FILENAME = f'{INPUT_FOLDER}/{INPUT_SUBFOLDER}/{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISE_FILENAME = f'{INPUT_FOLDER}/{NOISE_SUBFOLDER}/{NOISE_NAME}.{FILE_EXTENSION}'

RESULTS_RAW_FILENAME = f'{OUTPUT_FOLDER}/results_raw.csv'
RESULTS_SNR_FILENAME = f'{OUTPUT_FOLDER}/results_snr.csv'
RESULTS_MSE_FILENAME = f'{OUTPUT_FOLDER}/results_mse.csv'
INPUT_DATA_COPY_FILENAME = f'{OUTPUT_FOLDER}/{SIGNAL_NAME}.{FILE_EXTENSION}'
OUTPUT_DATA_FILENAME = f'{OUTPUT_FOLDER}/denoised_{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISY_DATA_FILENAME = f'{OUTPUT_FOLDER}/noisy_{SIGNAL_NAME}.{FILE_EXTENSION}'
NOISE_COPY_FILENAME = f'{OUTPUT_FOLDER}/noise_{NOISE_NAME}.{FILE_EXTENSION}'
REMAINING_NOISE_FILENAME = f'{OUTPUT_FOLDER}/remaining_noise.{FILE_EXTENSION}'


signal_sample_rate = 0
noise_sample_rate = 0


def init_mother_wavelets() -> list[str]:
    result = []

    daubechies = [ 'db2', 'db5', 'db8' ]
    result = np.append(result, daubechies)

    symlets = [ 'sym2', 'sym4' ]
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
    

def generate_white_noise(input_signal: ArrayLike, target_snr: float) -> ArrayLike:
    signal_power = np.mean(np.square(input_signal.astype(np.float64)))
    signal_power_db = 10*np.log10(signal_power)

    noise_power_db = signal_power_db - target_snr
    noise_power = np.power(10, noise_power_db/10)

    white_noise = np.random.normal(0, np.sqrt(noise_power), input_signal.size).astype(np.int16)

    return white_noise


def normalize(data: ArrayLike) -> ArrayLike:
    return data/np.power(2, 15)


def unnormalize(data: ArrayLike) -> ArrayLike:
    return (data*np.power(2, 15)).astype(np.int16)


def calculate_threshold(d1: ArrayLike, size: int, k: float = 0.8, m: float = 0.8) -> float:
    return k*m*np.median(np.abs(d1))/0.6745*np.sqrt(2*np.log(size))


def snr(signal: ArrayLike, noise: ArrayLike) -> float:
    signal_power_db = 10*np.log10(np.mean(np.square(signal.astype(np.float64))))
    noise_power_db = 10*np.log10(np.mean(np.square(noise.astype(np.float64))))

    return signal_power_db - noise_power_db


def mse(original_signal: ArrayLike, resulting_signal: ArrayLike) -> float:
    return np.mean(np.square(original_signal - resulting_signal))


def sort_dataset_by_snr(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values('output_snr', ascending=False)


def sort_dataset_by_mse(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values('output_mse', ascending=True)


def save_results(results: pd.DataFrame, input_data: ArrayLike, noise: ArrayLike) -> None:
    from pathlib import Path
    import os

    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, f'{OUTPUT_FOLDER}')

    Path(full_path).mkdir(parents=True, exist_ok=True)
    os.chmod(full_path, 0o770)

    results_by_snr = sort_dataset_by_snr(results)
    results_by_mse = sort_dataset_by_mse(results)

    best_config = results_by_snr.iloc[0]

    save_artifacts_from_configuration(
        input_data,
        noise,
        best_config['mother_wavelet'],
        best_config['local_max_level'],
        best_config['threshold_type'],
        best_config['k_coeff'],
        best_config['m_coeff']
    )

    results.to_csv(RESULTS_RAW_FILENAME, sep=';', decimal=',')
    results_by_snr.to_csv(RESULTS_SNR_FILENAME, sep=';', decimal=',')
    results_by_mse.to_csv(RESULTS_MSE_FILENAME, sep=';', decimal=',')

    print(f'Results saved to folder: {OUTPUT_FOLDER}')
    print(f'Best configuration: \n'
        + f'- Mother Wavelet: {best_config["mother_wavelet"]}\n'
        + f'- Wavelet transform level: {best_config["local_max_level"]}\n'
        + f'- Threshold type: {best_config["threshold_type"]}\n'
        + f'- \"k\" coefficient: {best_config["k_coeff"]}\n'
        + f'- \"m\" coefficient: {best_config["m_coeff"]}')


def save_artifacts_from_configuration(
    input_data: ArrayLike, 
    noise: ArrayLike,
    mother_wavelet: str, 
    max_level: int, 
    threshold_type: str,
    k_coeff: float,
    m_coeff: float
) -> dict:
    noisy_data = input_data + noise

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=max_level)

    wavelet_coefficients = transform[1:]
    coefficients_d1 = wavelet_coefficients[-1] # coefficients D1 from first level wavelet decomposition
    threshold = calculate_threshold(coefficients_d1, input_data.size, k_coeff, m_coeff) # threshold calculated for coefficients D1

    for index, coefficients in enumerate(wavelet_coefficients):
        wavelet_coefficients[index] = pywt.threshold(coefficients, value=threshold, mode=threshold_type)
    
    transform[1:] = wavelet_coefficients

    output_data = pywt.waverec(transform, mother_wavelet)
    output_data = output_data[:input_data.size].astype(np.int16)

    remaining_noise = output_data - input_data

    save_outputs_to_file(input_data, noise, noisy_data, output_data, remaining_noise)


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
    # mother_wavelets = [ 'db5' ]
    print(f'{mother_wavelets=}')

    global_max_level = GLOBAL_MAX_LEVEL
    # global_max_level = 6
    print(f'{global_max_level=}')

    threshold_types = init_threshold_types()
    # threshold_types = [ 'hard', 'soft' ]
    print(f'{threshold_types=}')

    k_coeffs = init_k_coeffs()
    # k_coeffs = [ 1 ]
    print(f'{k_coeffs=}')

    m_coeffs = init_m_coeffs()
    # m_coeffs = [ 1 ]
    print(f'{m_coeffs=}')

    print()

    data = get_input_data_from_file()[:signal_sample_rate*30]
    # noise = get_noise_from_file()
    noise = generate_white_noise(data, 5)[:signal_sample_rate*30]

    global noise_sample_rate

    noise_sample_rate = signal_sample_rate

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
        *(global_max_level)
        *len(threshold_types)
        *len(k_coeffs)
        *len(m_coeffs)
    )

    for wavelet in mother_wavelets:
        for level in range(1, global_max_level + 1):
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
    save_results(results, data, noise)


if __name__ == '__main__':
    main()
