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
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # width
h = 2160  # height
pp = 3.6 * um  # SLM pixel pitch
nzp = w
scaleX = 0.03

# depth map 조건
field_len = 1000e-3
near_depth = 800e-3
far_depth = 1200e-3

# depth quantization (깊이 계조)
DQ = 256  # 0 to 255

# unit depth
UD = -(far_depth - near_depth) / 256


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


class Depthmap:
    def __init__(self, RGBimg, Depthimg, f=1, angleX=0, angleY=0):
        self.zz = f
        self.imagein = np.asarray(Image.open(RGBimg))
        self.depthimg = np.asarray(Image.open(Depthimg))
        self.depthimg = self.depthimg[:, :, 1]
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.img_R = np.double(self.resizeimg(wvl_R, self.imagein[:, :, 0]))
        self.img_G = np.double(self.resizeimg(wvl_G, self.imagein[:, :, 1]))
        self.img_B = np.double(self.resizeimg(wvl_B, self.imagein[:, :, 2]))
        self.num_cpu = multiprocessing.cpu_count()
        self.num_point = [i for i in range(DQ)]
        self.num_point = np.array(self.num_point)



    def resizeimg(self, wvl, img):
        """RGB 파장에 맞게 원본 이미지를 리사이징 + zero padding"""
        w_n = int(w * (wvl_B / wvl))
        h_n = int(h * (wvl_B / wvl))
        img_new = np.zeros((h * 2, w * 2))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(h * 2 - h_n) // 2:(h * 2 + h_n) // 2, (w * 2 - w_n) // 2:(w * 2 + w_n) // 2] = im
        return img_new

    def Prop(self, color, image, dimg, amap, n):
        if color == 'green':
            wvl = wvl_G
            # image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            # image = self.img_B
        else:
            wvl = wvl_R
            # image = self.img_R
        # resize image
        imgs = image * amap
        imgs = np.flip(imgs, axis=0)
        zzz = UD * n + self.zz
        phase = np.random.random((2 * h, 2 * w)) * 2 * np.pi  # random phas
        ph = np.exp(1j * phase)
        ph *= imgs
        ps = scaleX / (2 * w)
        ch2 = ph * h_frsn(ps, ps, w + w, h + h, zzz, wvl)
        CH1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ch2)))
        result = CH1 * h_frsn(pp, pp, w + w, h + h, zzz, wvl)
        result = result[h // 2: (3 * h) // 2, w // 2: (3 * w) // 2]
        # result *= Refwave(wvl, zzz, self.thetaX, self.thetaY)
        return result

    def prop_R(self, n):
        wvl = wvl_R
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        return self.Prop('red', self.img_R, dimg, amap, n)

    def prop_G(self, n):
        wvl = wvl_G
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        return self.Prop('green', self.img_G, dimg, amap, n)

    def prop_B(self, n):
        wvl = wvl_B
        dimg = self.resizeimg(wvl, self.depthimg)
        amap = alphamap(dimg, n)
        return self.Prop('blue', self.img_B, dimg, amap, n)

    def parallelCal(self, color):
        if color == 'green':
            fun = self.prop_G
        elif color == 'blue':
            fun = self.prop_B
        else:
            fun = self.prop_R
        H = np.zeros((h, w), dtype='complex128')
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, DQ // self.num_cpu)])
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(fun, list(n))]
                cache = np.asarray(cache)
                print(n, ' depth done')
                for j in range(len(n)):
                    H += cache[j, :, :]
        return H

    def justCal(self, color):
        if color == 'green':
            wvl = wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_B
        else:
            wvl = wvl_R
            image = self.img_R
        dimg = self.resizeimg(wvl, self.depthimg)
        H = np.zeros((h, w), dtype='complex128')
        for i in range(DQ):
            amap = alphamap(dimg, i)
            H += self.Prop(color, image, dimg, amap, i)
            print(i, 'th layer done')
        return H

    def stepProp(self, color, img):
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        dimg = self.resizeimg(wvl, self.depthimg)
        i = 1
        plane = self.Prop(color, self.img_R, i)
        while i < DQ:
            i += 1
            amap = alphamap(dimg, i)
            plane += self.Prop(color, img, dimg, amap, i)
        return plane

    # functions
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


if __name__ == '__main__':
    fs = Depthmap('cubes.bmp', 'cubes_depth.bmp')
    dimg = fs.img_B
    print(dimg.shape)
    fs.getRGBImage(fs.parallelCal('red'), fs.parallelCal('green'), fs.parallelCal('blue'), '200916 depth map ex2.bmp')
    #fs.getRGBImage(fs.justCal('red'), fs.justCal('green'), fs.justCal('blue'), '200916 depthmap_cube1.bmp')
