import numpy as np
from pywt import Wavelet, dwt, idwt

def main():
    data = np.array([ 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 ], dtype=np.float64)

    haar = Wavelet('haar')
    db2 = Wavelet('db2')
    approx = data
    detail = list()

    approx, temp = dwt(approx, db2)
    detail.append(temp)
    print(approx)

    approx, temp = dwt(approx, db2)
    detail.append(temp)
    print(approx)

    approx, temp = dwt(approx, db2)
    detail.append(temp)
    print(approx)

    approx = idwt(approx, detail[2], db2)
    print(approx)

    approx = idwt(approx, detail[1], db2)
    print(approx)

    approx = idwt(approx, detail[0], db2)
    print(approx)


if __name__ == '__main__':
    main()
