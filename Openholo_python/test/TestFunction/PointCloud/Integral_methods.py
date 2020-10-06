import numpy as np
import matplotlib.pyplot as plt
import plyfile
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 1024  # 3840  # width
h = 1024  # 2160  # height
pp = 3.45 * 2 * um  # SLM pixel pitch
scaleXY = 0.001
scaleZ = 0.25


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True, cache=True)
def h_RS(x1, y1, z1, x2, y2, z2, wvl):
    r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
    t = (wvl * r) / (2 * pp)  # anti alliasing condition
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):
        h_r = np.sin(k(wvl) * r) / r ** 2
        h_i = np.cos(k(wvl) * r) / r ** 2
    else:
        h_r = 0
        h_i = 0
    return h_r, h_i


@njit(nogil=True)
def h_Frsn(x1, y1, z1, x2, y2, z2, wvl):
    """impulse response function of Fresnel propagation method"""
    z = z2 - z1
    r = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2*z)
    t = (wvl * z) / (2 * pp)
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
        h_r = np.cos(k(wvl) * r)
        h_i = np.sin(k(wvl) * r)
    else:
        h_r = 0
        h_i = 0
    return h_r, h_i


@njit(nogil=True)
def Conv(x1, y1, z1, z2, amp, wvl, method):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = (i - h / 2) * pp
            if method == 'RS':
                ch_r[i, j], ch_i[i, j] = h_RS(x1, y1, z1, x2, y2, z2, wvl)
            else:
                ch_r[i, j], ch_i[i, j] = h_Frsn(x1, y1, z1, x2, y2, z2, wvl)
    #print('point done')
    return (ch_r + 1j * ch_i) * amp


class Integral:
    def __init__(self, plypath, method='RS', f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.methods = method
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]
        self.num_point = np.array(self.num_point)

    def scale(self, wvl):
        if self.thetaX == 0 and self.thetaY == 0:
            return scaleXY
        else:
            return scaleXY# * (wvl_B / wvl)

    def Cal(self, n, color='red'):
        """Convolution"""
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        x0 = (self.plydata['x'][n] + self.z * np.tan(self.thetaX)) * self.scale(wvl)
        y0 = (self.plydata['y'][n] + self.z * np.tan(self.thetaY)) * self.scale(wvl)
        z0 = self.plydata['z'][n] * scaleZ
        amp = self.plydata[color][n] * (self.z / wvl)
        ch = Conv(x0, y0, z0, self.z, amp, wvl, self.methods)
        print(self.methods, 'methods ', n, ' th point ', color, ' Done')
        return ch

    def Conv_R(self, n):
        return self.Cal(n, 'red')

    def Conv_G(self, n):
        return self.Cal(n, 'green')

    def Conv_B(self, n):
        return self.Cal(n, 'blue')

    def CalHolo(self, color='red'):
        """Calculate hologram"""
        if color == 'green':
            func = self.Conv_G
        elif color == 'blue':
            func = self.Conv_B
        else:
            func = self.Conv_R
        print(self.num_cpu, " core Ready")
        ch = np.zeros((h, w), dtype='complex128')
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch

    # functions for encoding
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
        elif type == 'amplitude':
            arrin = np.copy(np.abs(arr))
        else:
            arrin = np.copy(arr)
        #arrin = np.float(arrin)
        arrin -= np.min(arrin)
        #arrin = arrin + np.abs(arrin)
        arrin = arrin / (np.max(arrin))
        return arrin

    def getRGBImage(self, R, G, B, fname, type='angle'):
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
        im = self.normalize(ch, 'angle')
        phase = fname + '_IM.bmp'
        plt.imsave(phase, im, cmap='gray')
        re = self.normalize(ch, 'amplitude')
        real = fname + '_RE.bmp'
        plt.imsave(real, re, cmap='gray')
        return im, re


if __name__ == '__main__':
    f1 = Integral('point_aperture.ply')                 # input ply file name
    ch = np.zeros((1024, 1024), dtype='complex128')     # define complex hologram map
    ch += f1.CalHolo('green')                           # Calculate hologram
    f2.getMonoImage(ch, '200921 GP model 2')            # save hologram