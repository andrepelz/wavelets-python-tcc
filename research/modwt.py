import numpy as np
from numpy.typing import ArrayLike
import pywt

def modwt_wavedec(data: ArrayLike, mother_wavelet: str, level: int = 1) -> ArrayLike:
    filters = pywt.Wavelet(mother_wavelet)

    scaling_filter = np.array(filters.dec_lo/np.sqrt(2))
    wavelet_filter = np.array(filters.dec_hi/np.sqrt(2))
    
    return Modwt.transform(data, level, scaling_filter, wavelet_filter)


def modwt_waverec(data: ArrayLike, mother_wavelet: str) -> ArrayLike:
    filters = pywt.Wavelet(mother_wavelet)

    scaling_filter = np.array(filters.rec_lo/np.sqrt(2))
    wavelet_filter = np.array(filters.rec_hi/np.sqrt(2))

    level = len(data) - 1
    
    return Modwt.inverse_transform(data, level, scaling_filter, wavelet_filter)


class Modwt:
    @staticmethod
    def transform(
            data: ArrayLike, 
            level: int,
            scaling_filter: ArrayLike, 
            wavelet_filter: ArrayLike) -> ArrayLike:
        result = np.array([data.astype(np.float64)])

        for current_level in np.arange(1, level + 1):
            result[0], detail_k = Modwt._step_transform(
                result[0], 
                current_level,
                scaling_filter,
                wavelet_filter)
            
            result = np.concatenate((result, [detail_k]))

        return result


    @staticmethod
    def inverse_transform(
            data: ArrayLike, 
            level: int,
            inverse_scaling_filter: ArrayLike, 
            inverse_wavelet_filter: ArrayLike) -> ArrayLike:
        result = data[0]

        for current_level in np.arange(level, 0, -1):
            result = Modwt._step_inverse_transform(
                result, 
                data[current_level], 
                current_level, 
                inverse_scaling_filter,
                inverse_wavelet_filter)
            
        return result

    @staticmethod
    def _step_transform(
            data_in: ArrayLike, 
            current_level: int,
            scaling_filter: ArrayLike, 
            wavelet_filter: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if scaling_filter.size != wavelet_filter.size:
            raise RuntimeError('Incompatible scaling and wavelet filters')
        
        filter_size = scaling_filter.size

        scaling_buffer = np.zeros(filter_size*current_level - (current_level - 1))
        wavelet_buffer = np.zeros(filter_size*current_level - (current_level - 1))

        padded_data_in = np.concatenate((data_in[-(scaling_buffer.size - 1):], data_in))

        scaling_buffer[::current_level] = scaling_filter
        wavelet_buffer[::current_level] = wavelet_filter

        approx_out = np.convolve(padded_data_in, scaling_buffer, mode='valid')
        detail_out = np.convolve(padded_data_in, wavelet_buffer, mode='valid')

        return (approx_out, detail_out)

    @staticmethod
    def _step_inverse_transform(
            approx_in: ArrayLike, 
            detail_in: ArrayLike, 
            current_level: int,
            inverse_scaling_filter: ArrayLike, 
            inverse_wavelet_filter: ArrayLike) -> ArrayLike:
        if inverse_scaling_filter.size != inverse_wavelet_filter.size:
            raise RuntimeError('Incompatible scaling and wavelet filters')
        
        filter_size = inverse_scaling_filter.size

        scaling_buffer = np.zeros(filter_size*current_level - (current_level - 1))
        wavelet_buffer = np.zeros(filter_size*current_level - (current_level - 1))

        padded_approx_in = np.concatenate((approx_in, approx_in[:(scaling_buffer.size - 1)]))
        padded_detail_in = np.concatenate((detail_in, detail_in[:(scaling_buffer.size - 1)]))

        scaling_buffer[::current_level] = inverse_scaling_filter
        wavelet_buffer[::current_level] = inverse_wavelet_filter

        approx_out = (np.convolve(padded_approx_in, scaling_buffer, mode='valid') 
                      + np.convolve(padded_detail_in, wavelet_buffer, mode='valid'))

        return approx_out
    