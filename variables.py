import numpy as np

MAX_LEVEL = 5

def init_transform_level() -> int:
    return MAX_LEVEL

def init_mother_wavelets() -> list:
    result = []

    daubechies = [ 'db2', 'db5', 'db8' ]
    result = np.append(result, daubechies)

    symlets = [ 'sym2', 'sym5', 'sym8' ]
    result = np.append(result, symlets)

    return result


def init_threshold_types() -> list:
    return [ 'hard', 'soft' ]


def init_k_coeffs() -> list:
    return [ 0.25, 0.5, 0.75, 1.0 ]


def init_m_coeffs() -> list:
    return [ 0.2, 0.4, 0.6, 0.8, 1.0 ]