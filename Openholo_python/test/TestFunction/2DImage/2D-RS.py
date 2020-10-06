import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from numba import njit

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 1024#3840            # width
h = 1024#2160            # height
pp = 3.6 * 2 * um       # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25

# source parameters
w_s = 1024              # source width
h_s = 1024              # height
ss = int(h_s * w_s)     # source size
ps = scaleXY / w_s      # source plane pixel pitch


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True)
def h_RS(x1, y1, z1, x2, y2, z2, wvl):
    """Impulse Response of R-S propagation"""
    zz = (z2 - z1) # * (wvl / wvl_B)  # point cloud 거리 좌표 보정
    r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + zz * zz)
    t = (wvl * r) / (2 * pp)  # anti aliasing condition
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):
        h_r = np.sin(k(wvl) * r)
        h_i = np.cos(k(wvl) * r)
    else:
        h_r = 0
        h_i = 0
    return h_r / (r * r), h_i / (r * r)


@njit(nogil=True)
def point_conv(n, image, z, wvl):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    x_s = int(n % w_s)
    y_s = int(n // w_s)
    amp = image[y_s, x_s]
    x1 = (x_s - w_s / 2) * ps  # source plane 좌표
    y1 = -(y_s - h_s / 2) * ps
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = -(i - h / 2) * pp
            re, im = h_RS(x1, y1, 0, x2, y2, z, wvl)
            ch_r[i, j] = re
            ch_i[i, j] = im
    ch_r = ch_r * amp
    ch_i = ch_i * amp
    # print(n, 'point done')
    return ch_r + 1j * ch_i


class RS:
    def __init__(self, imgpath, f=1):
        self.z = f  # Propagation distance
        self.imagein = np.asarray(Image.open(imgpath))
        self.num_cpu = 16 #multiprocessing.cpu_count()  # number of CPU
        self.img_R = np.double(self.imagein[:, :])
        self.img_G = np.double(self.imagein[:, :])
        self.img_B = np.double(self.imagein[:, :])
        self.num_point = []  # [i for i in range(ss)]     # number of point
        for i in range(ss):
            x_s = int(i % w_s)
            y_s = int(i // w_s)
            if self.img_R[y_s, x_s] == 0 and self.img_G[y_s, x_s] == 0 and self.img_B[y_s, x_s] == 0:
                pass
            else:
                self.num_point.append(i)

    def Cal(self, row, color='red'):
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
        ch = point_conv(row, image, self.z, wvl)
        print(row % w_s, ',', row//w_s ,' th point ', color, ' Done')
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
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, w // self.num_cpu)])
        print(len(count))
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch

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



def main():
    import time
    start = time.time()

    # aperture to GP = z0 : 295
    # CCD to GP = 5 mm
    # GP focal length = 100 mm
    ff = 1/(1/(100*mm) - 1/(295*mm))
    f1 = RS('test_apperture_F.bmp', f=ff+5*mm)
    f2 = RS('test_apperture_F.bmp', f=ff-5*mm)
    ch = np.zeros((1024, 1024), dtype='complex128')
    for n in f1.num_point:
        ch += f1.Cal(n, 'green')
        ch += f2.Cal(n, 'green')
    print(time.time() - start, ' is time')
    f2.getMonoImage(ch, '200921 gp modeling')

if __name__ == '__main__':
    main()