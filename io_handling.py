import numpy as np
import pandas as pd
import pywt
from numpy.typing import ArrayLike
from scipy.io import wavfile
from utils import calculate_threshold

def _get_signal_from_file(filename: str, file_extension: str) -> tuple[int, ArrayLike]:
    """
    Função que lê um sinal de um arquivo de áudio.

    Parâmetros
    ----------
    `filename` : `str`
        nome do arquivo de áudio a ser lido
    `file_extension` : `str`
        extensão do arquivo de áudio
        
    Retorna
    -------
    `tuple[int, ArrayLike]`
    """
    signal_sample_rate = None

    if file_extension == 'pcm':
        with open(filename, 'rb') as input_file:
            signal = np.fromfile(input_file, dtype=np.int16)
    elif file_extension == 'wav':
        signal_sample_rate, signal = wavfile.read(filename)
    else:
        raise ValueError('Unknown/Unsupported file extension.')

    return signal_sample_rate, signal


def _save_signal_to_file(data: ArrayLike, filename: str, file_extension: str, sample_rate: int = None):
    """
    Função que salva um sinal em um arquivo de áudio.

    Parâmetros
    ----------
    `data`: `ArrayLike`
        vetor com sinal a ser salvo
    `filename` : `str`
        nome do arquivo de áudio a ser salvo
    `file_extension` : `str`
        extensão do arquivo de áudio
    `sample_rate` : `int`
        taxa de amostragem do arquivo de áudio
    """
    if file_extension == 'pcm':
        with open(filename, 'wb') as output_file:
            output_file.write(data)
    elif file_extension == 'wav':
        if sample_rate is None:
            raise Exception("Sample rate not provided")
        wavfile.write(filename, sample_rate, data)
    else:
        raise ValueError('Unknown/Unsupported file extension.')


def get_input_data_from_file(filename: str, input_folder: str, file_extension: str) -> tuple[int, ArrayLike]:
    return _get_signal_from_file(f'{input_folder}/{filename}', file_extension)


def get_noise_from_file(filename: str, input_folder: str, file_extension: str) -> tuple[int, ArrayLike]:
    return _get_signal_from_file(f'{input_folder}/{filename}', file_extension)


def save_outputs_to_file(
        input_data: ArrayLike, 
        noise: ArrayLike, 
        noisy_data: ArrayLike, 
        output_data: ArrayLike, 
        remaining_noise: ArrayLike, 
        input_filename: str, 
        noise_filename: str, 
        output_folder: str, 
        file_extension: str,
        sample_rate: int = None) -> None:
    """
    Salva sinais de um teste em arquivos de áudio.

    Parâmetros
    ----------
    `input_data` : `ArrayLike`
        vetor com sinal de voz limpa de entrada
    `noise` : `ArrayLike`
        vetor com sinal de ruído
    `noisy_data` : `ArrayLike`
        vetor com sinal ruidoso
    `output_data` : `ArrayLike` 
        vetor do sinal de saída com ruído reduzido
    `remaining_noise` : `ArrayLike`
        vetor com ruído remanescente na saída
    `input_filename` : `str`
        nome do arquivo de áudio da voz limpa de entrada
    `noise_filename` : `str`
        nome do arquivo de áudio do ruído
    `output_folder` : `str`
        nome da pasta onde salvar os arquivos de saída
    `file_extension` : `str`
        extensão dos arquivos de áudio
    `sample_rate` : `int`
        taxa de amostragem dos arquivos de áudio
    """
    from pathlib import Path
    import os
    
    if len(input_data.shape) > 1:
        input_data = np.transpose(input_data)
        noise = np.transpose(noise)[0]
        noisy_data = np.transpose(noisy_data)
        output_data = np.transpose(output_data)
        remaining_noise = np.transpose(remaining_noise)

    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, output_folder)

    Path(full_path).mkdir(parents=True, exist_ok=True)

    _save_signal_to_file(input_data, f'{output_folder}/{input_filename}', file_extension, sample_rate)
    _save_signal_to_file(noise, f'{output_folder}/noise_{noise_filename}', file_extension, sample_rate)
    _save_signal_to_file(noisy_data, f'{output_folder}/noisy_{input_filename}', file_extension, sample_rate)
    _save_signal_to_file(output_data, f'{output_folder}/denoised_{input_filename}', file_extension, sample_rate)
    _save_signal_to_file(remaining_noise, f'{output_folder}/remaining_noise.{file_extension}', file_extension, sample_rate)


def _sort_dataset_by_snr(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena dataset por SNR de saída decrescente.

    Parâmetros
    ----------
    `dataset` : `DataFrame`
        dataset com resultados dos testes
        
    Retorna
    -------
    `DataFrame`
    """
    return dataset.sort_values('output_snr', ascending=False)


def _sort_dataset_by_mse(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena dataset por MSE de saída decrescente.

    Parâmetros
    ----------
    `dataset` : `DataFrame`
        dataset com resultados dos testes
        
    Retorna
    -------
    `DataFrame`
    """
    return dataset.sort_values('output_mse', ascending=True)


def _save_artifacts_from_configuration(
    input_data: ArrayLike, 
    noise: ArrayLike,
    mother_wavelet: str, 
    max_level: int, 
    threshold_type: str,
    k_coeff: float,
    m_coeff: float,
    input_filename: str, 
    noise_filename: str, 
    output_folder: str, 
    file_extension: str,
    sample_rate: int = None
) -> None:
    """
    Repete teste com melhor resultado e salva saída com ruído reduzido em um arquivo de áudio.

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
    `input_filename` : `str`
        nome do arquivo de áudio da voz limpa de entrada
    `noise_filename` : `str`
        nome do arquivo de áudio do ruído
    `output_folder` : `str`
        nome da pasta onde salvar os arquivos de saída
    `file_extension` : `str`
        extensão dos arquivos de áudio
    `sample_rate` : `int`
        taxa de amostragem dos arquivos de áudio
    """
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

    save_outputs_to_file(input_data, noise, noisy_data, output_data, remaining_noise, 
                         input_filename, noise_filename, output_folder, file_extension, sample_rate)


def save_results(
    results: pd.DataFrame, 
    input_data: ArrayLike, 
    noise: ArrayLike, 
    input_filename: str, 
    noise_filename: str, 
    output_folder: str, 
    file_extension: str,
    sample_rate: int = None
) -> None:
    """
    Função para salvar resultados do algoritmo em arquivos de saída.

    Parâmetros
    ----------
    `results` : `Dataframe`
        `Dataframe` da biblioteca pandas contendo os resultados de todos os testes
    `input_data` : `ArrayLike`
        vetor com sinal de voz limpa de entrada
    `noise` : `ArrayLike`
        vetor com sinal de ruído
    `input_filename` : `str`
        nome do arquivo de áudio da voz limpa de entrada
    `noise_filename` : `str`
        nome do arquivo de áudio do ruído
    `output_folder` : `str`
        nome da pasta onde salvar os arquivos de saída
    `file_extension` : `str`
        extensão dos arquivos de áudio
    `sample_rate` : `int`
        taxa de amostragem dos arquivos de áudio
    """
    from pathlib import Path
    import os

    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, output_folder)

    Path(full_path).mkdir(parents=True, exist_ok=True)
    os.chmod(full_path, 0o770)

    results_by_snr = _sort_dataset_by_snr(results)
    results_by_mse = _sort_dataset_by_mse(results)

    best_config = results_by_snr.iloc[0]

    _save_artifacts_from_configuration(
        input_data,
        noise,
        best_config['mother_wavelet'],
        best_config['local_max_level'],
        best_config['threshold_type'],
        best_config['k_coeff'],
        best_config['m_coeff'], 
        input_filename, 
        noise_filename, 
        output_folder, 
        file_extension, 
        sample_rate
    )

    results.to_csv(f'{output_folder}/results_raw.csv', sep=';', decimal=',')
    results_by_snr.to_csv(f'{output_folder}/results_snr.csv', sep=';', decimal=',')
    results_by_mse.to_csv(f'{output_folder}/results_mse.csv', sep=';', decimal=',')

    print(f'Results saved to folder: {output_folder}')
    print(f'Best configuration: \n'
        + f'- Mother Wavelet: {best_config["mother_wavelet"]}\n'
        + f'- Wavelet transform level: {best_config["local_max_level"]}\n'
        + f'- Threshold type: {best_config["threshold_type"]}\n'
        + f'- \"k\" coefficient: {best_config["k_coeff"]}\n'
        + f'- \"m\" coefficient: {best_config["m_coeff"]}')
