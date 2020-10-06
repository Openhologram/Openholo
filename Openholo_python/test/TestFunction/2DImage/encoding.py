import numpy as np
import matplotlib.pyplot as plt
import scipy


class Encoding:
    def fft(self, f):
        return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(f)))

    def ifft(self, f):
        return scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.fftshift(f)))

    def SSB(self, ch):
        """single side band encoding"""
        height, width = ch.shape
        a = np.zeros((height, width), dtype='complex128')
        CH = self.fft(ch)  # fourier transformed image
        CH = CH[height // 4: (height * 3) // 4, :]
        a[0:height // 2, :] = CH
        a[height // 2:, :] = np.conj(CH)
        return self.ifft(a)

    def normalize(self, arr, type='angle'):
        """normalize"""
        if type == 'phase':
            arrin = np.copy(np.imag(arr))
        elif type == 'real':
            arrin = np.copy(np.real(arr))
        elif type == 'angle':
            arrin = np.copy(np.angle(arr))
            arrin = (arrin + np.pi) / (2 * np.pi)
            return arrin
        elif type == 'amplitude':
            arrin = np.copy(np.abs(arr))
        else:
            arrin = np.copy(arr)
        # arrin = np.float(arrin)
        arrin -= np.min(arrin)
        # arrin = arrin + np.abs(arrin)
        arrin = arrin / np.max(arrin)
        return arrin

    def getRGBImage(self, R, G, B, fname, type='phase'):
        """Get RGB image"""
        h, w = R.shape
        img = np.zeros((h, w, 3))
        img[:, :, 0] = self.normalize(R, type)
        img[:, :, 1] = self.normalize(G, type)
        img[:, :, 2] = self.normalize(B, type)
        plt.imsave(fname, img)
        return img

    def getMonoImage(self, ch, fname):
        """Get Single channel image"""
        im = self.normalize(ch, 'phase')
        phase = fname + '_IM.bmp'
        plt.imsave(phase, im, cmap='gray')
        re = self.normalize(ch, 'real')
        real = fname + '_RE.bmp'
        plt.imsave(real, re, cmap='gray')
        return im, re
