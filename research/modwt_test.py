import numpy as np
import pywt
from modwt import Modwt

def main():
    array = np.arange(32, dtype=np.int16)

    wavelet = pywt.Wavelet('db5')

    scaling_filter = wavelet.dec_lo
    wavelet_filter = wavelet.dec_hi

    transform = Modwt._step_transform(array, 1, scaling_filter/np.sqrt(2), wavelet_filter/np.sqrt(2))
    approx, detail = Modwt._step_transform(transform[0], 2, scaling_filter/np.sqrt(2), wavelet_filter/np.sqrt(2))
    inverse_transform = Modwt._step_inverse_transform(approx, detail, 2, scaling_filter/np.sqrt(2), wavelet_filter/np.sqrt(2))
    inverse_transform = Modwt._step_inverse_transform(inverse_transform, transform[1], 1, scaling_filter/np.sqrt(2), wavelet_filter/np.sqrt(2))

    # print(transform)
    print(inverse_transform)
    print(np.round(inverse_transform).astype(np.int16))

    old_transform = np.copy(transform)
    old_inverse_transform = np.copy(inverse_transform)

    print()
    print('===========================================================')
    print()

    transform = Modwt.transform(array, 6, scaling_filter, wavelet_filter)
    # print(transform)
    inverse_transform = Modwt.inverse_transform(transform, 6, scaling_filter, wavelet_filter)
    # inverse_transform = Modwt._step_inverse_transform(transform[0], transform[1], 1, scaling_filter/np.sqrt(2), wavelet_filter/np.sqrt(2))
    print(inverse_transform)
    print(np.round(inverse_transform).astype(np.int16))