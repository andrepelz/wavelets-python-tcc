import numpy as np
from numpy.typing import ArrayLike


def update_noise_to_target_snr(
        noise: ArrayLike, 
        signal: ArrayLike, 
        target_snr_db: float) -> ArrayLike:
    """
    Ajusta ganho de um sinal de ruído para uma dada SNR alvo.

    Parâmetros
    ----------
    `noise` : `ArrayLike`
        vetor com o sinal de ruído
    `signal` : `ArrayLike`
        vetor com o sinal de interesse limpo
    `target_snr_db` : `ArrayLike`
        valor de SNR alvo para o sinal ruidoso

    Retorna
    -------
    `ArrayLike`
    """
    result = np.copy(noise)

    signal_power = _signal_power(signal)
    noise_power = _signal_power(noise)

    signal_power_db = 10*np.log10(signal_power)

    target_noise_power_db = signal_power_db - target_snr_db
    target_noise_power = np.power(10, target_noise_power_db/10)

    target_power_ratio = target_noise_power/noise_power

    result = np.sqrt(target_power_ratio)*result

    return result.astype(np.int16)


def normalize(data: ArrayLike) -> ArrayLike:
    """
    Normaliza um vetor com inteiros 16 bit entre os valores -1 e 1.

    Parâmetros
    ----------
    `data` : `ArrayLike`
        vetor com o sinal de interesse

    Retorna
    -------
    `ArrayLike`
    """
    return data/np.power(2, 15)


def unnormalize(data: ArrayLike) -> ArrayLike:
    """
    Converte um vetor normalizado entre -1 e 1 em um vetor de inteiros 16 bit.

    Parâmetros
    ----------
    `data` : `ArrayLike`
        vetor com o sinal de interesse

    Retorna
    -------
    `ArrayLike`
    """
    return (data*np.power(2, 15)).astype(np.int16)


def calculate_threshold(d1: ArrayLike, size: int, k: float = 0.8, m: float = 0.8) -> float:
    """
    Calcula o valor de threshold para um determinado conjunto de coeficientes detail D1, k e m.

    Parâmetros
    ----------
    `d1` : `ArrayLike`
        coeficientes de detail de primeiro nível D1 de uma Transformada Wavelet
    `size` : `int`
        tamanho do sinal original
    `k` : `float`
        coeficiente k
    `m` : `float`
        coeficiente m

    Retorna
    -------
    `float`
    """
    return k*m*np.median(np.abs(d1))/0.6745*np.sqrt(2*np.log(size))


def _signal_power(data: ArrayLike) -> float:
    """
    Calcula a potência de um sinal.

    Parâmetros
    ----------
    `data` : `ArrayLike`
        vetor com o sinal de interesse

    Retorna
    -------
    `float`
    """
    return np.mean(np.square(data.astype(np.float64)))


def snr(signal: ArrayLike, noise: ArrayLike) -> float:
    """
    Calcula a SNR entre dois sinais.

    Parâmetros
    ----------
    `signal` : `ArrayLike`
        vetor com o sinal de interesse
    `noise` : `ArrayLike`
        vetor com o sinal de ruído

    Retorna
    -------
    `float`
    """
    signal_power_db = 10*np.log10(_signal_power(signal))
    noise_power_db = 10*np.log10(_signal_power(noise))

    return signal_power_db - noise_power_db


def mse(original_signal: ArrayLike, resulting_signal: ArrayLike) -> float:
    """
    Calcula o MSE entre dois sinais.

    Parâmetros
    ----------
    `original_signal` : `ArrayLike`
        vetor com o sinal original
    `resulting_signal` : `ArrayLike`
        vetor com o sinal resultante

    Retorna
    -------
    `float`
    """
    return np.mean(np.square(original_signal - resulting_signal))