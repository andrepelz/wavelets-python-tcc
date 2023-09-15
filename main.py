import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike
import pywt


GLOBAL_MAX_LEVEL = 6
SIGNAL_SAMPLE_RATE = 8000
SIGNAL_FRAME_SIZE = int(0.02*SIGNAL_SAMPLE_RATE) # 20 ms
SIGNAL_FRAME_OVERLAP = int(SIGNAL_FRAME_SIZE*0.5) # 50% overlap
# SIGNAL_FRAME_OVERLAP = 0

SIGNAL_NAME = 'speech'
INPUT_DATA_FILENAME = f'{SIGNAL_NAME}.pcm'
OUTPUT_DATA_FILENAME = f'denoised_{SIGNAL_NAME}.pcm'
NOISY_DATA_FILENAME = f'noisy_{SIGNAL_NAME}.pcm'
# NOISE_FILENAME = 'white_noise.pcm'
NOISE_FILENAME = f'{SIGNAL_NAME}_noise.pcm'


def init_mother_wavelets() -> list[str]:
    result = []

    daubechies = [ 'db4', 'db8', 'db16', 'db24', 'db32' ]
    result = np.append(result, daubechies)

    symlets = [ 'sym2', 'sym4', 'sym8', 'sym16' ]
    result = np.append(result, symlets)

    coiflets = [ 'coif2', 'coif4', 'coif8' ]
    result = np.append(result, coiflets)

    return result


def init_threshold_types() -> list[str]:
    return [ 'hard', 'soft', 'garrote' ]


def get_input_data_from_file() -> ArrayLike:
    filename = INPUT_DATA_FILENAME
    with open(filename, 'rb') as input_file:
        signal = np.fromfile(input_file, dtype=np.int16)

    return signal


def get_noise_from_file() -> ArrayLike:
    filename = NOISE_FILENAME
    with open(filename, 'rb') as input_file:
        noise = np.fromfile(input_file, dtype=np.int16)

    return noise


def median_absolute_deviation(data: ArrayLike) -> float:
    return np.median(np.abs(data - np.mean(data)))


def calculate_threshold(d1: ArrayLike, size: int, k: float = 0.5) -> float:
    return k*median_absolute_deviation(d1)/0.6745*np.sqrt(2*np.log(size))


def snr(signal: ArrayLike, noise: ArrayLike) -> float:
    signal_power = np.mean(np.square(signal.astype(np.float64), dtype=np.float64))
    noise_power = np.mean(np.square(noise.astype(np.float64), dtype=np.float64))

    return 10*np.log10(signal_power/noise_power)


def mse(original_signal: ArrayLike, resulting_signal: ArrayLike) -> float:
    """
    Returns the mean squared error (MSE) between a given signal and its resulting signal

    Parameters 
    ----------
    original_signal : array_like
        Array object of the original signal
    resulting_signal : array_like 
        Array object of the resulting signal (transformed, predicted, etc.)

    Returns
    -------
    float
        The mean squared error value between the original signal and its resulting signal
    """
    return np.mean(np.square(original_signal.astype(np.float64) - resulting_signal.astype(np.float64)))


def normalize(data: ArrayLike) -> ArrayLike:
    return data/norm(data)


def frame_based_denoising_kernel(
        noisy_data, 
        mother_wavelet, 
        local_max_level, 
        threshold_type
    ):
    output_data = np.zeros(noisy_data.size, dtype=noisy_data.dtype)

    for frame_start in np.arange(0, noisy_data.size, SIGNAL_FRAME_SIZE - SIGNAL_FRAME_OVERLAP): # frame based approach (algorithm applied independently to 400 sample frames)
        frame_end = frame_start + SIGNAL_FRAME_SIZE
        if frame_start + SIGNAL_FRAME_SIZE > noisy_data.size:
            frame_end = noisy_data.size 

        frame = noisy_data[frame_start:frame_end]
        actual_frame_size = frame_end - frame_start

        transform = pywt.wavedec(frame, mother_wavelet, level=local_max_level)

        coefficients_d1 = transform[-1] # coefficients D1 from first level wavelet decomposition
        threshold = calculate_threshold(coefficients_d1, frame.size, 0.6) # threshold calculated for coefficients D1

        wavelet_coefficients = transform[1:]

        for index, coefficients in enumerate(wavelet_coefficients):
            wavelet_coefficients[index] = pywt.threshold(coefficients, value=threshold, mode=threshold_type)
        
        transform[1:] = wavelet_coefficients

        reconstructed_frame = pywt.waverec(transform, mother_wavelet)
        reconstructed_frame *= np.hamming(reconstructed_frame.size)

        # reconstructed frame is sliced as it might have one more sample than the original as a side effect of the IDWT
        # this may happen because usually, the IDWT algorithm can only be applied to signals with an even number of samples
        # therefore, when working with an odd sample sized signal, a padding sample is added as to not affect the algorithm
        output_data[frame_start:frame_end] += np.array(reconstructed_frame[:actual_frame_size], dtype=np.int16)

    return output_data


def evaluate_noise_reduction_algorithm(
    input_data: ArrayLike, 
    noise: ArrayLike,
    mother_wavelet: str, 
    max_level: int, 
    threshold_type: str
) -> tuple[tuple[str, int, str], list[float]]:
    local_max_level = min(max_level, pywt.dwt_max_level(input_data.size, mother_wavelet))
    # local_max_level = min(max_level, pywt.dwt_max_level(SIGNAL_FRAME_SIZE, mother_wavelet))

    if local_max_level < max_level: # if local max level is reduced, then further calculations will be duplicated redundant
        return (None, None)
    
    noisy_data = input_data + noise

    # input_mse = mse(normalize(input_data), normalize(noisy_data))
    input_mse = mse(input_data, noisy_data)
    input_snr = snr(input_data, noise)

    # output_data = frame_based_denoising_kernel(noisy_data, mother_wavelet, local_max_level, threshold_type)

    transform = pywt.wavedec(noisy_data, mother_wavelet, level=local_max_level)

    coefficients_d1 = transform[-1] # coefficients D1 from first level wavelet decomposition
    threshold = calculate_threshold(coefficients_d1, input_data.size, 0.6) # threshold calculated for coefficients D1

    wavelet_coefficients = transform[1:]

    for index, coefficients in enumerate(wavelet_coefficients):
        wavelet_coefficients[index] = pywt.threshold(coefficients, value=threshold, mode=threshold_type)
    
    transform[1:] = wavelet_coefficients

    output_data = pywt.waverec(transform, mother_wavelet)
    output_data = output_data[:input_data.size].astype(np.int16)

    remaining_noise = output_data - input_data

    # output_mse = mse(normalize(input_data), normalize(output_data))
    output_mse = mse(input_data, output_data)
    output_snr = snr(input_data, remaining_noise)

    key = (mother_wavelet, local_max_level, threshold_type)
    values = { 
        'input_snr': input_snr, 
        'output_snr': output_snr, 
        'input_mse': input_mse, 
        'output_mse': output_mse
    }

    if local_max_level == 2:
        filename = NOISY_DATA_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(noisy_data)

        filename = OUTPUT_DATA_FILENAME
        with open(filename, 'wb') as output_file:
            output_file.write(output_data)

        filename = 'remaining_noise.pcm'
        with open(filename, 'wb') as output_file:
            output_file.write(remaining_noise)

    return (key, values)


def main():
    # mother_wavelets = init_mother_wavelets()
    mother_wavelets = ['db8']
    print(f'{mother_wavelets=}')

    # global_max_level = GLOBAL_MAX_LEVEL
    global_max_level = 5
    print(f'{global_max_level=}')

    # threshold_types = init_threshold_types()
    threshold_types = ['soft']
    print(f'{threshold_types=}')

    data = get_input_data_from_file()
    noise = get_noise_from_file()
    noise = noise[:data.size]


    results = dict()

    for wavelet in mother_wavelets:
        for level in range(global_max_level):
            for thresh_type in threshold_types:
                key, values = evaluate_noise_reduction_algorithm(data, noise, wavelet, level + 1, thresh_type)

                if key != None:
                    results[key] = values

    for key, value in results.items():
        print(key, value)


if __name__ == '__main__':
    main()
