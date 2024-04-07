import numpy as np
from numpy.typing import ArrayLike

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