import numpy as np
import plyfile
import multiprocessing
import scipy

from concurrent.futures import ProcessPoolExecutor
from numba import njit
import matplotlib.pyplot as plt

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # width
h = 2160  # height
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25
ps = scaleXY / w  # source plane pixel pitch (sampling rate)
Wr = pp * w  # Receiver plane width
Ws = scaleXY  # Source plane width


# inline function
@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True, cache=True)
def nzps(zz, wvl):
    p = (wvl * zz) / np.sqrt(4 * ps * ps - wvl * wvl)
    return int((1 / ps) * (Wr / 2 - Ws / 2 + p) + 1)

nzp = 2 * w

@njit
def limits(u, z, wvl):
    # u is delta u
    return 1/(wvl * np.sqrt((2 * u * z)**2 + 1))

@njit
def asm_kernel(wvl, z):
    deltax = 1 / (w * pp * 3)     # sampling period
    a = np.zeros((w * 3, w *3))        # real part
    b = np.zeros((w*3, w*3))        # imaginary part
    delx = limits(deltax, z, wvl)
    dely = limits(deltax, z, wvl)
    for i in range(w*3):
        for j in range(w*3):
            fx = ((i - w*3/2) * deltax)
            fy = -((j - w*3/2) * deltax)
            if -delx < fx < delx and -dely < fy < dely: # band limiting
            #if (fx * fx + fy * fy) < (1 / (wvl * wvl)):
                a[j, i] = np.cos(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
                b[j, i] = np.sin(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
    print(z, 'kernel ready')
    return a + 1j * b


@njit(nogil=True, cache=True)
def Refwave(wvl, r, thetax, thetay):
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = (j - w / 2) * pp
            y = -(i - h / 2) * pp
            a[i, j] = np.cos(-k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
            b[i, j] = np.sin(-k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
    return a / r - 1j * (b / r)



class ASM:
    """Angular spectrum method FFT propagation"""
    def __init__(self, plypath, f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]
        self.num_point = np.array(self.num_point)

    def sfft(self, f):
        return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(f)))

    def sifft(self, f):
        return scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.fftshift(f)))

    def Cal(self, n, color='red'):
        """FFT calcuation"""
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        x0 = self.plydata['x'][n] * 2
        y0 = self.plydata['y'][n] * 2
        zz = (self.z - self.plydata['z'][n] * scaleZ)
        N = nzp
        W = N + w  # 2 * w
        # point map
        pmap = np.zeros((W, W))
        p = np.int((x0 + 1) * (w/2))
        q = np.int((1 - y0) * (w/2))
        pmap[q + N // 2, p + N // 2] = 1
        print(p,', ', q, 'th p map done')
        amp = self.plydata[color][n] * pmap
        amp = self.sfft(amp)
        kernel = asm_kernel(wvl, zz)
        ch = self.sifft(amp * kernel)
        del amp
        ch = ch[(W - h) // 2: (W + h) // 2, (W - w) // 2: (W + w) // 2]
        #ch = ch * Refwave(wvl, zz, self.thetaX, self.thetaY) * self.plydata[color][n]
        print(n, ' point', color, ' is done')
        return ch

    def FFT_R(self, n):
        return self.Cal(n, 'red')

    def FFT_G(self, n):
        return self.Cal(n, 'green')

    def FFT_B(self, n):
        return self.Cal(n, 'blue')

    def CalHolo(self, color='red'):
        """Calculate hologram"""
        if color == 'green':
            func = self.FFT_G
        elif color == 'blue':
            func = self.FFT_B
        else:
            func = self.FFT_R
        print(self.num_cpu, " core Ready")
        ch = np.zeros((h, w), dtype='complex128')
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
        # print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch

    # functions for encoding
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
        # arrin = np.float(arrin)
        arrin -= np.min(arrin)
        # arrin = arrin + np.abs(arrin)
        arrin = arrin / np.max(arrin)
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
        im = self.normalize(ch, 'phase')
        phase = fname + '_IM.bmp'
        plt.imsave(phase, im, cmap='gray')
        re = self.normalize(ch, 'real')
        real = fname + '_RE.bmp'
        plt.imsave(real, re, cmap='gray')
        return im, re


if __name__ == '__main__':
    f = ASM('point_3.ply')                              # input file name
    f.getMonoImage(f.CalHolo('green'), '200917 ASM2')   # Calculate Hologram and save