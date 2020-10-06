import numpy as np
import matplotlib.pyplot as plt
import plyfile
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# unit parameters
mm = 1e-3
um = mm * mm
nm = um * mm


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True, cache=True)
def h_RS(x1, y1, z1, x2, y2, z2, wvl, pp):
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
def h_Frsn(x1, y1, z1, x2, y2, z2, wvl, pp):
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
def Conv(x1, y1, z1, z2, amp, h, w, pp, wvl, method):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = (i - h / 2) * pp
            if method == 'RS':
                ch_r[i, j], ch_i[i, j] = h_RS(x1, y1, z1, x2, y2, z2, wvl, pp)
            else:
                ch_r[i, j], ch_i[i, j] = h_Frsn(x1, y1, z1, x2, y2, z2, wvl, pp)
    return (ch_r + 1j * ch_i) * amp


class Propagation:
    """
    Get Fringe pattern by Point Cloud data(.ply)

    Parameters
    plypath : .ply file path
    angle : phase shift angle
    Red, Green, Blue : wavelength of RGB color
    scale : scaling factor

    http://openholo.org/
    """
    def __init__(self, plypath, method='RS', propagation_distance=1, angleX=0, angleY=0, Red=639*nm, Green=525*nm, Blue=463*nm,
                 SLM_width=3840, SLM_height=2160, scaleXY=0.03, scaleZ=0.25, pixel_pitch=3.6*um, multicore=True):
        self.z = propagation_distance  # Propagation distance
        self.methods = method
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.wvl_R = Red
        self.wvl_G = Green
        self.wvl_B = Blue
        self.w = SLM_width
        self.h = SLM_height
        self.pp = pixel_pitch
        self.scaleXY = scaleXY
        self.scaleZ = scaleZ
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        if multicore:
            self.num_cpu = multiprocessing.cpu_count()
        else:
            self.num_cpu = 1
        self.num_point = [i for i in range(len(self.plydata))]
        self.num_point = np.array(self.num_point)


    def Cal(self, n, color='red'):
        """Convolution"""
        if color == 'green':
            wvl = self.wvl_G
        elif color == 'blue':
            wvl = self.wvl_B
        else:
            wvl = self.wvl_R
        x0 = (self.plydata['x'][n] + self.z * np.tan(self.thetaX)) * self.scaleXY
        y0 = (self.plydata['y'][n] + self.z * np.tan(self.thetaY)) * self.scaleXY
        z0 = self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n] * (self.z / wvl)
        ch = Conv(x0, y0, z0, self.z, amp, self.h, self.w, self.pp, wvl, self.methods)
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
        ch = np.zeros((self.h, self.w), dtype='complex128')
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
    def fft(self, f):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))

    def ifft(self, f):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))

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
