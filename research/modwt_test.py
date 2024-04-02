import numpy as np
import pywt
from modwt import Modwt, modwt_wavedec, modwt_waverec

def main():
    array = np.arange(1024, dtype=np.int16)

    wavelet = pywt.Wavelet('db5')

    dec_scaling_filter = wavelet.dec_lo/np.sqrt(2)
    dec_wavelet_filter = wavelet.dec_hi/np.sqrt(2)
    rec_scaling_filter = wavelet.rec_lo/np.sqrt(2)
    rec_wavelet_filter = wavelet.rec_hi/np.sqrt(2)

    transform = Modwt._step_transform(array, 1, dec_scaling_filter, dec_wavelet_filter)
    approx, detail = Modwt._step_transform(transform[0], 2, dec_scaling_filter, dec_wavelet_filter)
    inverse_transform = Modwt._step_inverse_transform(approx, detail, 2, rec_scaling_filter, rec_wavelet_filter)
    inverse_transform = Modwt._step_inverse_transform(inverse_transform, transform[1], 1, rec_scaling_filter, rec_wavelet_filter)

    # print(transform)
    print(inverse_transform)
    print(np.round(inverse_transform).astype(np.int16))

    print()
    print('===========================================================')
    print()

    transform = Modwt.transform(array, 6, dec_scaling_filter, dec_wavelet_filter)
    inverse_transform = Modwt.inverse_transform(transform, 6, rec_scaling_filter, rec_wavelet_filter)

    print(inverse_transform)
    print(np.round(inverse_transform).astype(np.int16))

    print()
    print('===========================================================')
    print()

    mother_wavelet = 'db1'

    transform = modwt_wavedec(array, mother_wavelet, 5)
    inverse_transform = modwt_waverec(transform, mother_wavelet)

    print(inverse_transform)
    print(np.round(inverse_transform).astype(np.int16))

if __name__ == '__main__':
    main()