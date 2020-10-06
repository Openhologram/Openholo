# input image
import numpy as np
from numba import njit, prange, jit
import matplotlib.pyplot as plt
from PIL import Image
from 2DImage.encoding import Encoding
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w_s = 1920  # source width
h_s = 1080  # height
w = 3840  # SLM width
h = 2160
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25
ps = scaleXY / w_s  # source plane pixel pitch
ss = int(h_s * w_s)  # source size
slm_s = h*w     # SLM size



@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True)
def h_Fresnel(x1, y1, x2, y2, z, wvl):
    """impulse response function of Fresnel propagation method"""
    r = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2*z)
    # t = (wvl * z) / (2 * pp)  안티 엘리어싱 삭제
    h_r = np.cos(k(wvl) * r)
    h_i = np.sin(k(wvl) * r)
    return h_r, h_i


@njit(nogil=True)
def point_conv(n, image, z, wvl):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    x_s = int(n % w_s)
    y_s = int(n // w_s)
    if image[y_s, x_s] == 0:
        return None
    amp = image[y_s, x_s]
    x1 = (x_s - w_s / 2) * ps  # source plane 좌표
    y1 = -(y_s - h_s / 2) * ps
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = -(i - h / 2) * pp
            re, im = h_Fresnel(x1, y1, x2, y2, z, wvl)
            ch_r[i, j] = re
            ch_i[i, j] = im
    ch_r = ch_r * amp
    ch_i = ch_i * amp
    print(n, 'point done')
    return ch_r + 1j * ch_i


@njit(nogil=True)
def Conv(image, z, wvl):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    for i in range(int(ss)):
        x_s = int(i % w_s)
        y_s = int(i // w_s)
        amp = image[y_s, x_s]
        x1 = (x_s - w_s/2) * ps  # source plane 좌표
        y1 = -(y_s - h_s/2) * ps
        im = np.zeros((h,w))
        re = np.zeros((h,w))
        for j in range(int(slm_s)):
            x_r = j % w
            y_r = j // w
            x2 = (x_r - w/2) * pp
            y2 = (y_r - h/2) * pp
            real, imag = h_Fresnel(x1, y1, x2, y2, z, wvl)
            re[y_r, x_r] = real * amp
            im[y_r, x_r] = imag * amp
        ch_r += re
        ch_i += im
        print(x_s, ', ',y_s, 'point done')
    return ch_r + 1j * ch_i


class Frsn(Encoding):
    def __init__(self, imgpath, f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.img = np.asarray(Image.open(imgpath))
        self.img_R = np.double(self.img[:, :, 0])
        self.img_G = np.double(self.img[:, :, 1])
        self.img_B = np.double(self.img[:, :, 2])
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = []  #[i for i in range(ss)]     # number of point
        for i in range(ss):
            x_s = int(i % w_s)
            y_s = int(i // w_s)
            if self.img_R[y_s, x_s]==0 and self.img_G[y_s, x_s]==0 and self.img_B[y_s, x_s]==0:
                pass
            else:
                self.num_point.append(i)


    def Cal(self, n, color='red'):
        """Convolution"""
        # ch = np.zeros((h, w), dtype='complex128')
        if color == 'green':
            wvl = wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_B
        else:
            wvl = wvl_R
            image = self.img_R
        ch = point_conv(n, image, self.z, wvl)
        print(n % w_s, ',', n//w_s ,' th point ', color, ' Done')
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
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, ss // self.num_cpu)])
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch

def main():
    import time
    start = time.time()
    f = Frsn('aperture2.bmp')
    red = f.CalHolo()
    print(time.time() - start, ' is time')
    g = f.CalHolo('green')
    b = f.CalHolo('blue')
    ch = f.getRGBImage(red, g, b, '200824_2D Frsn_aperture.bmp')
    plt.imshow(np.real(ch))
    plt.imsave('diceimage2.bmp', ch)

if __name__ == '__main__':
    main()