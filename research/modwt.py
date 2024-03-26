import numpy as np
from numpy.typing import ArrayLike

class Modwt:
    @staticmethod
    def transform(
            data: ArrayLike, 
            level: int,
            scaling_filter: ArrayLike, 
            wavelet_filter: ArrayLike):
        result = np.array([data.astype(np.float64)])

        for current_level in np.arange(1, level + 1):
            result[0], detail_k = Modwt._step_transform(
                result[0], 
                current_level,
                scaling_filter/np.sqrt(2),
                wavelet_filter/np.sqrt(2))
            
            result = np.concatenate((result, [detail_k]))

        return result


    @staticmethod
    def inverse_transform(
            data: ArrayLike, 
            level: int,
            inverse_scaling_filter: ArrayLike, 
            inverse_wavelet_filter: ArrayLike):
        result = data[0]

        for current_level in np.arange(level, 0, -1):
            result = Modwt._step_inverse_transform(
                result, 
                data[current_level], 
                current_level, 
                inverse_scaling_filter/np.sqrt(2),
                inverse_wavelet_filter/np.sqrt(2))
            
        return result

    @staticmethod
    def _step_transform(
            data_in: ArrayLike, 
            current_level: int,
            step_scaling_filter: ArrayLike, 
            step_wavelet_filter: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if step_scaling_filter.size != step_wavelet_filter.size:
            raise RuntimeError('Incompatible scaling and wavelet filters')

        input_size = data_in.size

        approx_out = np.array([], dtype=data_in.dtype)
        detail_out = np.array([], dtype=data_in.dtype)

        for t in np.arange(data_in.size):
            k = t

            buffer_step = np.power(2, current_level - 1)
            input_buffer = np.array([data_in[k]], dtype=data_in.dtype)

            for _ in np.arange(1, step_scaling_filter.size):
                k = (k - buffer_step + input_size)%input_size
                input_buffer = np.concatenate([ input_buffer, [data_in[k]] ])

            item_approx_out = np.sum(input_buffer*step_scaling_filter)
            item_detail_out = np.sum(input_buffer*step_wavelet_filter)

            approx_out = np.concatenate([ approx_out, [item_approx_out] ])
            detail_out = np.concatenate([ detail_out, [item_detail_out] ])

        return (approx_out, detail_out)

    @staticmethod
    def _step_inverse_transform(
            approx_in: ArrayLike, 
            detail_in: ArrayLike, 
            current_level: int,
            step_inverse_scaling_filter: ArrayLike, 
            step_inverse_wavelet_filter: ArrayLike) -> ArrayLike:
        if step_inverse_scaling_filter.size != step_inverse_wavelet_filter.size:
            raise RuntimeError('Incompatible scaling and wavelet filters')

        input_size = approx_in.size

        approx_out = np.array([], dtype=approx_in.dtype)

        for t in np.arange(approx_in.size):
            k = t

            buffer_step = np.power(2, current_level - 1)
            input_approx_buffer = np.array([approx_in[k]], dtype=approx_in.dtype)
            input_detail_buffer = np.array([detail_in[k]], dtype=approx_in.dtype)

            for _ in np.arange(1, step_inverse_scaling_filter.size):
                k = (k + buffer_step)%input_size
                input_approx_buffer = np.concatenate([ input_approx_buffer, [approx_in[k]] ])
                input_detail_buffer = np.concatenate([ input_detail_buffer, [detail_in[k]] ])

            item_approx_out = (np.sum(input_approx_buffer*step_inverse_scaling_filter) 
                               + np.sum(input_detail_buffer*step_inverse_wavelet_filter))

            approx_out = np.concatenate([ approx_out, [item_approx_out] ])

        return approx_out
    