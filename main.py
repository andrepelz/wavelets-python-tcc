import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from spectrum import Spectrum
from wavelets import DaubechiesD4 as db4, Haar

def impulso_unitario(n, dtype=np.float64):
    y = np.zeros(n.shape, dtype)
    
    zero, = np.where(n == 0)
    
    for index in zero:
        y[index] = 1

    return y

def degrau_unitario(n, dtype=np.float64):
    y = np.zeros(n.shape, dtype)
    
    zero, = np.where(n == 0)
    
    for index in zero:
        y[index:] = 1

    return y

def pkt_test():
    data = np.array(
        [32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34],
        dtype=np.float64
    )

    transform = Haar.packet_transform(data)

    # result = transform.get_full_transform()

    basis = transform.get_best_basis()
    result = np.array([], dtype=np.float64)

    for item in basis:
        result = np.append(result, item.data)

    # recovery = Haar.inverse_packet_transform(result)

    np.set_printoptions(precision=2)
    print(data)
    print(result)
    # print(recovery)

    return

def spectrum_and_transform_test(*, cls=db4):
    n = np.arange(0, np.pi, np.pi/1024)

    wave_1 = np.sin( 4*np.pi*n)
    wave_2 = np.sin(16*np.pi*n)

    data = wave_1 + wave_2

    wavelet_transform = cls.transform(data)
    power_spectrum = Spectrum.spectral_calc(wavelet_transform)
    inverse_transform = cls.inverse_transform(wavelet_transform)

    n_spectrum = np.arange(0, power_spectrum.size)

    plt.plot(n, data)
    plt.plot(n, inverse_transform)
    plt.show()

    plt.plot(n_spectrum, power_spectrum)
    plt.stem(n_spectrum, power_spectrum, linefmt='red')
    plt.show()

    component_1, component_2 = Spectrum.reconstruct_signal(wavelet_transform, cls=cls)

    plt.plot(n, component_1)
    plt.plot(n, component_2)
    plt.show()

def noise_reduction_test(): # terrible, horrible, not good, very bad
    """not that terrible, could be better"""
    def calculate_threshold(data: ArrayLike, k: float = 0.5, decomposition_level: int = 1):
        # std_deviation = np.std(data)
        return k*np.std(data)*np.sqrt(2*np.log2(data.size))/np.log2((decomposition_level + 1))

    def apply_threshold(data: ArrayLike, threshold: float, thresh_type='hard', beta: float = 0.1):
        result = np.copy(data)
        
        for index, sample in enumerate(result):
            if np.abs(sample) < threshold and thresh_type != 'improved': # sample under threshold check
                result[index] = 0
            elif thresh_type == 'soft': # sample over threshold, do nothing on hard thresh, subtract threshold value on soft thresh
                result[index] = sample - threshold if sample > 0 else sample + threshold
            elif thresh_type == 'improved':
                if np.abs(sample) < threshold:
                    result[index] = beta*sample
                else:
                    partial_value = np.sqrt(np.square(sample) - np.square(threshold))
                    result[index] = partial_value if sample > 0 else -partial_value

        return result


    filename = 'sine.pcm'
    with open(filename, 'rb') as input_file:
        signal = np.fromfile(input_file, dtype=np.int16)
    filename = 'white_noise.pcm'
    with open(filename, 'rb') as input_file:
        noise = np.fromfile(input_file, dtype=np.int16)


    # sample_rate = 40000
    # n = np.arange(0.2*sample_rate) # 0.2 second signal
    # signal = np.int16( np.power(2, 14)*np.sin(2*n*np.pi*100/sample_rate) )
    # noise = np.int16( np.power(2, 10)*np.sin(2*n*np.pi*3000/sample_rate) )


    data = np.add(signal, noise, dtype=np.int16)
    n = np.arange(data.size)

    depth = 2

    plt.plot(n, data)
    plt.show()

    transform = db4.transform(data, depth)

    plt.plot(n, transform)
    plt.show()

    inverse_transform = transform
    # inverse_transform = db4.inverse_transform(transform, depth).astype(np.int16)

    # start = transform.size//np.power(2, depth)
    # end = start*2
    # inverse_transform[start:end] = apply_threshold(inverse_transform[start:end])

    for i in range(depth):
        start = transform.size//np.power(2, depth - i)
        end = start*2
        level = depth - i
        frequency_band = inverse_transform[start:end]
        inverse_transform[start:end] = apply_threshold(
            data=frequency_band, 
            threshold=calculate_threshold(frequency_band, k=0.8, decomposition_level=level), 
            # threshold=np.power(2, 14), 
            thresh_type='improved'
        )
        # plt.plot(n, inverse_transform)
        # plt.show()
        step = db4.inverse_transform(inverse_transform[:end], 1)
        inverse_transform[:end] = step

    # inverse_transform = db4.inverse_transform(inverse_transform, depth)

    inverse_transform = inverse_transform.astype(data.dtype)

    plt.plot(n, data, 'r')
    plt.plot(n, inverse_transform, 'g')
    plt.show()

    filename = 'noisy_sine.pcm'
    with open(filename, 'wb') as output_file:
        output_file.write(data)

    filename = 'poorly_denoised_signal.pcm'
    with open(filename, 'wb') as output_file:
        output_file.write(inverse_transform)
    
    return

def main():
    noise_reduction_test()
    # pkt_test()
    # spectrum_and_transform_test(cls=Haar)

    return

    
if __name__ == '__main__':
    main()