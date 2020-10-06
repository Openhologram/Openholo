import numpy as np
import plyfile
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from numba import njit
from functions.encoding import Encoding

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
Nzp = w  # zero padding number
Wtotal = int(Nzp + w)  # total width


# inline function
@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True)
def pointmap(x, y):
    pmap = np.zeros((Wtotal, Wtotal))
    p = np.int((x + 1) * (w / 2))
    q = np.int((-y + 1) * (w / 2))
    pmap[q + Nzp // 2, p + Nzp // 2] = 1
    print(p, ',', q)
    return pmap


@njit(nogil=True, cache=True)
def h_frsn(pixel):
    re = np.zeros((Wtotal, Wtotal))
    im = np.zeros((Wtotal, Wtotal))
    for i in range(Wtotal):
        for j in range(Wtotal):
            x = (i - Wtotal / 2) * pixel
            y = (j - Wtotal / 2) * pixel
            re[j, i] = np.cos(np.pi * (x * x + y * y))
            im[j, i] = np.sin(np.pi * (x * x + y * y))
    return re + 1j * im


@njit(nogil=True)
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


class FrsnFFT(Encoding):
    """fresnel FFT propagation"""
    def __init__(self, plypath, f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]
        self.u_slm = h_frsn(pp)  # * 1/(wvl * zz)
        self.u_obj = h_frsn(ps)

    def scale(self, wvl):
            return scaleXY * (wvl_B / wvl)

    def Cal(self, n, color='red'):
        """FFT calcuation"""
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        x0 = self.plydata['x'][n]
        y0 = self.plydata['y'][n]
        zz = self.z - self.plydata['z'][n] * scaleZ
        amp = self.plydata[color][n]
        pmap = pointmap(x0, y0)
        #pmap = np.flip(pmap, axis=0)
        u0 = pmap * (self.u_obj ** (1 / (wvl * zz))) * amp
        u0 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(u0)))
        ch = (np.exp(1j * k(wvl) * zz) / (1j * wvl * zz)) * (self.u_slm ** (1 / (wvl * zz)))
        ch = ch * u0
        ch = ch[(Wtotal - h) // 2: (Wtotal + h) // 2, (Wtotal - w) // 2: (Wtotal + w) // 2]
        # ch = ch * Refwave(wvl, zz, self.thetaX, self.thetaY)
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
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch

