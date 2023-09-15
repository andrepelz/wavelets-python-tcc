import numpy as np
from numpy.typing import ArrayLike

from wavelets import IWaveletTransform

class Spectrum:
    def __new__(cls, *args, **kwargs):
        if cls is Spectrum:
            raise TypeError(f'This class is designed to only provide static methods. Instantiating a {cls.__name__} object is prohibited.')
        
    def spectral_calc(data : ArrayLike) -> ArrayLike:
        start = 0
        end = 1

        power_spectrum = []

        sample_size = data.size

        while end <= sample_size:
            power = 0

            power = np.sum(data[start:end]*data[start:end])
            power_spectrum.append(power)

            start = end
            end *= 2
        
        result = np.array(power_spectrum, dtype=np.float64)

        return result
    
    def copy_bands(source : ArrayLike, start_band : int, end_band : int) -> ArrayLike:
        start = 0
        end = 1
        curr_band = 0

        result = np.zeros(source.size, dtype=source.dtype)

        result[0] = source[0]

        while end <= source.size:
            if curr_band in range(start_band, end_band + 1):
                result[start:end] = source[start:end]

            start = end
            end *= 2
            curr_band += 1

        return result

    def reconstruct_signal(data: ArrayLike, *, cls: IWaveletTransform) -> ArrayLike:
        num_bands = np.log2(data.size).astype(np.int16)

        first_half = Spectrum.copy_bands(data, 1, num_bands//2)
        first_half = cls.inverse_transform(first_half)

        second_half = Spectrum.copy_bands(data, num_bands//2 + 1, num_bands)
        second_half = cls.inverse_transform(second_half)

        return (first_half,  second_half)
