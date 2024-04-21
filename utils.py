import numpy as np
from numpy.typing import ArrayLike


def update_noise_to_target_snr(
        noise: ArrayLike, 
        signal: ArrayLike, 
        target_snr_db: float):
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
    return data/np.power(2, 15)


def unnormalize(data: ArrayLike) -> ArrayLike:
    return (data*np.power(2, 15)).astype(np.int16)


def calculate_threshold(d1: ArrayLike, size: int, k: float = 0.8, m: float = 0.8) -> float:
    return k*m*np.median(np.abs(d1))/0.6745*np.sqrt(2*np.log(size))


def _signal_power(data: ArrayLike):
    return np.mean(np.square(data.astype(np.float64)))


def snr(signal: ArrayLike, noise: ArrayLike) -> float:
    signal_power_db = 10*np.log10(_signal_power(signal))
    noise_power_db = 10*np.log10(_signal_power(noise))

    return signal_power_db - noise_power_db


def mse(original_signal: ArrayLike, resulting_signal: ArrayLike) -> float:
    return np.mean(np.square(original_signal - resulting_signal))