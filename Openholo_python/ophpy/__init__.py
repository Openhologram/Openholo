import numpy as np
import matplotlib.pyplot as plt

__name__ = "ophpy"
__version__ = "0.0.4"


def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))


def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))


def SSB(ch):
    """single side band encoding"""
    height, width = ch.shape
    a = np.zeros((height, width), dtype='complex128')
    CH = fft(ch)  # fourier transformed image
    CH = CH[height // 4: (height * 3) // 4, :]
    a[0:height // 2, :] = CH
    a[height // 2:, :] = np.conj(CH)
    return ifft(a)


def normalize(arr, type='angle'):
    """normalize"""
    if type == 'phase':
        arrin = np.copy(np.imag(arr))
    elif type == 'real':
        arrin = np.copy(np.real(arr))
    elif type == 'angle':
        arrin = np.copy(np.angle(arr))
    elif type == 'amplitude':
        arrin = np.copy(np.abs(arr))
    else:
        arrin = np.copy(arr)
    # arrin = np.float(arrin)
    arrin -= np.min(arrin)
    # arrin = arrin + np.abs(arrin)
    arrin = arrin / np.max(arrin)
    return arrin


def getRGBImage(R, G, B, fname, type='phase'):
    """Get RGB image"""
    h, w = R.shape
    img = np.zeros((h, w, 3))
    img[:, :, 0] = normalize(R, type)
    img[:, :, 1] = normalize(G, type)
    img[:, :, 2] = normalize(B, type)
    plt.imsave(fname, img)
    return img


def getMonoImage(ch, fname):
    """Get Single channel image"""
    im = normalize(ch, 'phase')
    phase = fname + '_IM.bmp'
    plt.imsave(phase, im, cmap='gray')
    re = normalize(ch, 'real')
    real = fname + '_RE.bmp'
    plt.imsave(real, re, cmap='gray')
    return im, re
