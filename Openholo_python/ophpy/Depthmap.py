import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from PIL import Image

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm


@njit
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True, cache=True)
def h_frsn(pixel_pitch_x, pixel_pitch_y, nx, ny, zz, wvl):
    re = np.zeros((ny, nx))
    im = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = (i - nx / 2) * pixel_pitch_x
            y = (j - ny / 2) * pixel_pitch_y
            re[j, i] = np.cos((np.pi / (wvl * zz)) * (x * x + y * y))
            im[j, i] = np.sin((np.pi / (wvl * zz)) * (x * x + y * y))
    return re + 1j * im


@njit(nogil=True)
def alphamap(depthmap, n):
    """ extract alpha map """
    shape = depthmap.shape
    ww = shape[1]
    hh = shape[0]
    amap = np.zeros((hh, ww))
    for i in range(ww):
        for j in range(hh):
            if depthmap[j, i] == n:
                amap[j, i] = 1
    return amap


@njit
def refwave(wvl, wr, hr, z, pp, thetaX, thetaY):
    a = np.zeros((hr, wr))
    b = np.zeros((hr, wr))
    for i in range(hr):
        for j in range(wr):
            x = (j - wr / 2) * pp
            y = -(i - hr / 2) * pp
            a[i, j] = np.cos(k(wvl) * (x * np.sin(thetaX) + y * np.sin(thetaY)))
            b[i, j] = np.sin(k(wvl) * (x * np.sin(thetaX) + y * np.sin(thetaY)))
    return (a / z) + 1j * (b / z)


class Propagation:
    """
    Get Fringe pattern by 2D Depth map image and RGB color image

    Parameters
    imgpath : image path
    f : propagation length
    angle : phase shift angle
    Red, Green, Blue : wavelength
    scale : scaling factor

    Depth map parameters
    field_len = 1000e-3
    near_depth = 800e-3
    far_depth = 1200e-3

    Depth quantization (깊이 계조)
    DQ = 256  # 0 to 255

    Unit depth
    UD = -(far_depth - near_depth) / 256
    http://openholo.org/
    """
    def __init__(self, RGBimg, Depthimg, f=1, angleX=0, angleY=0,
                 Red=639*nm, Green=525*nm, Blue=463*nm, SLM_width=3840, multicore=True,
                 SLM_height=2160, scale=0.03, pixel_pitch=3.6*um, zeropadding=3840,
                 field_length=1000e-3, near_depth=800e-3, far_depth=1200e-3, DepthQuantization=256):
        self.zz = f
        self.imagein = np.asarray(Image.open(RGBimg))
        self.depthimg = np.asarray(Image.open(Depthimg))
        self.depthimg = self.depthimg[:, :, 1]
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.wvl_R = Red
        self.wvl_G = Green
        self.wvl_B = Blue
        self.w = SLM_width
        self.h = SLM_height
        self.pp = pixel_pitch
        self.scale = scale
        self.UD = -(far_depth - near_depth) / 256
        self.DQ = DepthQuantization
        self.nzp = zeropadding
        self.img_R = np.double(self.resizeimg(self.wvl_R, self.imagein[:, :, 0]))
        self.img_G = np.double(self.resizeimg(self.wvl_G, self.imagein[:, :, 1]))
        self.img_B = np.double(self.resizeimg(self.wvl_B, self.imagein[:, :, 2]))
        if multicore:
            self.num_cpu = multiprocessing.cpu_count()
        else:
            self.num_cpu = 1
        self.num_point = [i for i in range(self.DQ)]
        self.num_point = np.array(self.num_point)

    def resizeimg(self, wvl, img):
        """RGB 파장에 맞게 원본 이미지를 리사이징 + zero padding"""
        w_n = int(self.w * (self.wvl_B / wvl))
        h_n = int(self.h * (self.wvl_B / wvl))
        img_new = np.zeros((self.h * 2, self.w * 2))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        img_new[(self.h * 2 - h_n) // 2:(self.h * 2 + h_n) // 2, (self.w * 2 - w_n) // 2:(self.w * 2 + w_n) // 2] = im
        return img_new

    def Prop(self, color, image, amap, n):
        if color == 'green':
            wvl = self.wvl_G
        elif color == 'blue':
            wvl = self.wvl_B
        else:
            wvl = self.wvl_R
        # resize image
        imgs = image * amap
        imgs = np.flip(imgs, axis=0)
        zzz = self.UD * n + self.zz
        phase = np.random.random((2 * self.h, 2 * self.w)) * 2 * np.pi  # random phase
        ph = np.exp(1j * phase)
        ph *= imgs
        ps = self.scale / (2 * self.w)
        ch2 = ph * h_frsn(ps, ps, self.w + self.w, self.h + self.h, zzz, wvl)
        CH1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ch2)))
        result = CH1 * h_frsn(self.pp, self.pp, self.w + self.w, self.h + self.h, zzz, wvl)
        result = result[self.h // 2: (3 * self.h) // 2, self.w // 2: (3 * self.w) // 2]
        result *= refwave(wvl, self.w, self.h, zzz, self.pp, self.thetaX, self.thetaY)
        return result

    def prop_R(self, n):
        wvl = self.wvl_R
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        print(n, ' th layer done')
        return self.Prop('red', self.img_R, amap, n)

    def prop_G(self, n):
        wvl = self.wvl_G
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        print(n, ' th layer done')
        return self.Prop('green', self.img_G, amap, n)

    def prop_B(self, n):
        wvl = self.wvl_B
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        print(n, ' th layer done')
        return self.Prop('blue', self.img_B, amap, n)

    def parallelCal(self, color):
        if color == 'green':
            fun = self.prop_G
        elif color == 'blue':
            fun = self.prop_B
        else:
            fun = self.prop_R
        H = np.zeros((self.h, self.w), dtype='complex128')
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, self.DQ // self.num_cpu)])
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(fun, list(n))]
                cache = np.asarray(cache)
                print(n, ' depth done')
                for j in range(len(n)):
                    H += cache[j, :, :]
        return H


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


