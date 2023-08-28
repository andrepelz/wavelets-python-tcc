import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from packet_transform import *


SQRT_2 = 1.4142135623730950488016887242097
SQRT_3 = 1.7320508075688772935274463415059

class IWaveletTransform(ABC):
    """Interface for a Mother Wavelet class and its transforms"""
    @abstractmethod
    def transform(data : ArrayLike, level : int) -> ArrayLike:
        pass
    
    @abstractmethod
    def inverse_transform(data : ArrayLike, level : int) -> ArrayLike:
        pass
    
    @abstractmethod
    def packet_transform(data : ArrayLike, level : int) -> ArrayLike:
        pass

    @abstractmethod
    def inverse_packet_transform(data : ArrayLike, level : int) -> ArrayLike:
        pass

class DaubechiesD4(IWaveletTransform):
    SCALING_COEFS = np.array( [
        (1 + SQRT_3)/(4*SQRT_2),
        (3 + SQRT_3)/(4*SQRT_2),
        (3 - SQRT_3)/(4*SQRT_2),
        (1 - SQRT_3)/(4*SQRT_2)
    ], dtype=np.float64 )

    WAVELET_COEFS = np.array( [
         SCALING_COEFS[3],
        -SCALING_COEFS[2],
         SCALING_COEFS[1],
        -SCALING_COEFS[0]
    ], dtype=np.float64 )

    INVERSE_SCALING_COEFS = np.array( [
        SCALING_COEFS[2],
        WAVELET_COEFS[2],
        SCALING_COEFS[0],
        WAVELET_COEFS[0]
    ], dtype=np.float64 )

    INVERSE_WAVELET_COEFS = np.array( [
        SCALING_COEFS[3],
        WAVELET_COEFS[3],
        SCALING_COEFS[1],
        WAVELET_COEFS[1]
    ], dtype=np.float64 )

    @staticmethod
    def transform(data: ArrayLike, level: int = np.inf) -> ArrayLike:
        if data.size >= 4:
            if data.size//np.power(2, level - 1) < 4 and level != np.inf:
                raise RuntimeError(f'Transform level provided is too deep! Please provide a smaller level')
            
            result = np.copy(data).astype(np.float64)
            end = data.size

            curr_level = 0

            while end >= 4 and curr_level < level:
                result[:end] = DaubechiesD4._step_transform(result[:end])
                end = end//2
                curr_level += 1

            return result
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 4 data points')
        
    @staticmethod
    def inverse_transform(data: ArrayLike, level: int = np.inf) -> ArrayLike:
        if data.size >= 4:
            result = np.copy(data).astype(np.float64)
            end = 4 if level == np.inf else data.size//np.power(2, level - 1)

            if end < 4:
                raise RuntimeError(f'Transform level provided is too deep! Please provide a smaller level')

            while end <= data.size:
                result[:end] = DaubechiesD4._step_inverse_transform(result[:end])
                end = end*2

            return result
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 4 data points')
    
    @staticmethod
    def packet_transform(data : ArrayLike, level : int = np.inf) -> ArrayLike:
        if data.size >= 4:
            if data.size//np.power(2, level - 1) < 4:
                raise RuntimeError(f'Transform level provided is too deep! Please provide a smaller level')
            
            buffer = np.copy(data)
            end = data.size
            section = end

            curr_level = 0

            transform_tree = TransformTree()
            transform_tree.add_node(buffer)

            while section >= 4 and curr_level < level:
                for start in range(0, end, section):
                    limit = start + section
                    middle = (start + limit)//2
                    buffer[start:limit] = DaubechiesD4._step_transform(buffer[start:limit])

                    transform_tree.add_node( buffer[ start : middle ] )
                    transform_tree.add_node( buffer[ middle : limit ] )

                curr_level += 1
                section = section//2

            return transform_tree
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 4 data points')

    @staticmethod
    def inverse_packet_transform(data : ArrayLike, level : int = np.inf) -> ArrayLike:
        raise NotImplementedError(f'The function {DaubechiesD4.inverse_packet_transform.__name__} is currently under construction.')

    def _step_transform(data : ArrayLike) -> ArrayLike:
        even = np.zeros(data.size//2, dtype=data.dtype)
        odd  = np.zeros(data.size//2, dtype=data.dtype)

        for i in range(0, data.size - 3, 2):
            start = i
            end = i + 4
            even[i//2] = np.sum( data[start:end]*DaubechiesD4.SCALING_COEFS )
            odd [i//2] = np.sum( data[start:end]*DaubechiesD4.WAVELET_COEFS )

        # special case for final values at finite datasets
        even[-1] = (
            data[-2]*DaubechiesD4.SCALING_COEFS[0]
          + data[-1]*DaubechiesD4.SCALING_COEFS[1]
          + data [0]*DaubechiesD4.SCALING_COEFS[2]
          + data [1]*DaubechiesD4.SCALING_COEFS[3]
        )

        odd [-1] = (
            data[-2]*DaubechiesD4.WAVELET_COEFS[0]
          + data[-1]*DaubechiesD4.WAVELET_COEFS[1]
          + data [0]*DaubechiesD4.WAVELET_COEFS[2]
          + data [1]*DaubechiesD4.WAVELET_COEFS[3] 
        )

        output = np.append(even, odd)

        return output
    
    def _step_inverse_transform(data : ArrayLike) -> ArrayLike:
        even = np.zeros(data.size//2, dtype=data.dtype)
        odd = np.zeros(data.size//2, dtype=data.dtype)

        interleaved_data = DaubechiesD4._interleave(data[:data.size//2], data[data.size//2:])

        # special case for first values at finite datasets
        even[0] = (
            interleaved_data[-2]*DaubechiesD4.INVERSE_SCALING_COEFS[0]
          + interleaved_data[-1]*DaubechiesD4.INVERSE_SCALING_COEFS[1]
          + interleaved_data [0]*DaubechiesD4.INVERSE_SCALING_COEFS[2]
          + interleaved_data [1]*DaubechiesD4.INVERSE_SCALING_COEFS[3]
        )

        odd [0] = (
            interleaved_data[-2]*DaubechiesD4.INVERSE_WAVELET_COEFS[0]
          + interleaved_data[-1]*DaubechiesD4.INVERSE_WAVELET_COEFS[1]
          + interleaved_data [0]*DaubechiesD4.INVERSE_WAVELET_COEFS[2]
          + interleaved_data [1]*DaubechiesD4.INVERSE_WAVELET_COEFS[3]
        )

        for i in range(0, interleaved_data.size - 3, 2):
            start = i
            end = i + 4
            even[i//2 + 1] = np.sum( interleaved_data[start:end]*DaubechiesD4.INVERSE_SCALING_COEFS )
            odd [i//2 + 1] = np.sum( interleaved_data[start:end]*DaubechiesD4.INVERSE_WAVELET_COEFS )

        output = DaubechiesD4._interleave(even, odd)

        return output

    def _interleave(left : ArrayLike, right : ArrayLike) -> ArrayLike:
        interleaved_data = np.zeros(2*left.size, dtype=left.dtype)
        interleaved_data [::2] = left
        interleaved_data[1::2] = right

        return interleaved_data
        
class Haar(IWaveletTransform):
    SCALING_COEFS = np.array( [
         0.5,
         0.5
    ], dtype=np.float64 )

    WAVELET_COEFS = np.array( [
         0.5,
        -0.5
    ], dtype=np.float64 )

    INVERSE_SCALING_COEFS = np.array( [
         1,
         1
    ], dtype=np.float64 )

    INVERSE_WAVELET_COEFS = np.array( [
         1,
        -1
    ], dtype=np.float64 )

    @staticmethod
    def transform(data : ArrayLike, level : int = np.inf) -> ArrayLike:
        if data.size >= 2:
            result = np.copy(data)
            end = data.size

            while end >= 2:
                result[:end] = Haar._step_transform(result[:end])
                end = end//2

            return result
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 2 data points')
        
    @staticmethod
    def inverse_transform(data : ArrayLike, level : int = np.inf) -> ArrayLike:
        if data.size >= 2:
            result = np.copy(data)
            end = 2

            while end <= data.size:
                result[:end] = Haar._step_inverse_transform(result[:end])
                end = end*2

            return result
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 4 data points')
    
    @staticmethod
    def packet_transform(data : ArrayLike, level : int = np.inf) -> TransformTree:
        if data.size >= 2:
            buffer = np.copy(data)
            end = data.size
            section = end

            transform_tree = TransformTree()
            transform_tree.add_node(buffer)

            while section >= 2:
                for start in range(0, end, section):
                    limit = start + section
                    middle = (start + limit)//2
                    buffer[start:limit] = Haar._step_transform(buffer[start:limit])

                    transform_tree.add_node( buffer[ start : middle ] )
                    transform_tree.add_node( buffer[ middle : limit ] )

                section = section//2

            return transform_tree
        else:
            raise RuntimeError(f'Sample size of data provided is too small! Please provide a sample of at least 2 data points')

    @staticmethod
    def inverse_packet_transform(data : ArrayLike, level : int = np.inf) -> ArrayLike:
        raise NotImplementedError(f'The function {Haar.inverse_packet_transform.__name__} is currently under construction.')

    def _step_transform(data : ArrayLike) -> ArrayLike:
        even = np.zeros(data.size//2, dtype=data.dtype)
        odd = np.zeros(data.size//2, dtype=data.dtype)

        for i in range(0, data.size, 2):
            start = i
            end = i + 2
            even[i//2] = np.sum( data[start:end]*Haar.SCALING_COEFS )
            odd [i//2] = np.sum( data[start:end]*Haar.WAVELET_COEFS )

        output = np.append(even, odd)

        return output
    
    def _step_inverse_transform(data : ArrayLike) -> ArrayLike:
        even = np.zeros(data.size//2, dtype=data.dtype)
        odd = np.zeros(data.size//2, dtype=data.dtype)

        interleaved_data = Haar._interleave(data[:data.size//2], data[data.size//2:])

        for i in range(0, interleaved_data.size, 2):
            start = i
            end = i + 2
            even[i//2] = np.sum( interleaved_data[start:end]*Haar.INVERSE_SCALING_COEFS )
            odd [i//2] = np.sum( interleaved_data[start:end]*Haar.INVERSE_WAVELET_COEFS )

        output = Haar._interleave(even, odd)

        return output

    def _interleave(left : ArrayLike, right : ArrayLike) -> ArrayLike:
        interleaved_data = np.zeros(2*left.size, dtype=left.dtype)
        interleaved_data [::2] = left
        interleaved_data[1::2] = right

        return interleaved_data
    