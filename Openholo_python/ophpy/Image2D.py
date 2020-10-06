import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm

@njit
def k(wvl):
    return (np.pi * 2) / wvl


@njit
def limits(u, z, wvl):
    # u is delta u
    return (1/wvl) * np.sqrt((2 * u * z)**2 + 1)


@njit
def asm_kernel(wvl, z, w, h, pp):
    deltax = 1 / (w * pp * 4)     # sampling period
    deltay = 1 / (h * pp * 4)
    a = np.zeros((h*3, w*3))        # real part
    b = np.zeros((h*3, w*3))        # imaginary part
    delx = limits(deltax, z, wvl)
    dely = limits(deltay, z, wvl)
    for i in range(w*3):
        for j in range(h*3):
            fx = ((i - w*(3/2)) * deltax)
            fy = -((j - h*(3/2)) * deltay)
            if -delx < fx < delx and -dely < fy < dely:
                a[j, i] = np.cos(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
                b[j, i] = np.sin(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
    print(z, 'kernel ready')
    return a + 1j * b


@njit(nogil=True, cache=True)
def h_frsn(pixel_pitch_x, pixel_pitch_y, nx, ny, zz, wvl):
    re = np.zeros((ny, nx))
    im = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = (i - nx / 2) * pixel_pitch_x
            y = -(j - ny / 2) * pixel_pitch_y
            re[j, i] = np.cos((np.pi / (wvl * zz)) * (x * x + y * y))
            im[j, i] = np.sin((np.pi / (wvl * zz)) * (x * x + y * y))
    return re + 1j * im


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
    Get Fringe pattern by 2D image

    Parameters
    imgpath : image path
    f : propagation length
    angle : phase shift angle
    Red, Green, Blue : wavelength
    scale : scaling factor

    http://openholo.org/
    """
    def __init__(self, imgpath, propagation_distance=1, angleX=0, angleY=0, Red=639*nm, Green=525*nm, Blue=463*nm,
                 SLM_width=3840, SLM_height=2160, scale=0.03, pixel_pitch=3.6*um, zeropadding=3840):
        self.zz = propagation_distance
        self.imagein = np.asarray(Image.open(imgpath))
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.wvl_R = Red
        self.wvl_G = Green
        self.wvl_B = Blue
        self.w = SLM_width
        self.h = SLM_height
        self.pp = pixel_pitch
        self.scale = scale
        self.nzp = zeropadding
        self.img_R = np.double(self.resizeimg(self.wvl_R, self.imagein[:, :, 0]))
        self.img_G = np.double(self.resizeimg(self.wvl_G, self.imagein[:, :, 1]))
        self.img_B = np.double(self.resizeimg(self.wvl_B, self.imagein[:, :, 2]))
        self.img_r = np.double(self.zeropadding(self.imagein[:,:,0], self.w * 2, self.h * 2))
        self.img_g = np.double(self.zeropadding(self.imagein[:,:,1], self.w * 2, self.h * 2))
        self.img_b = np.double(self.zeropadding(self.imagein[:,:,2], self.w * 2, self.h * 2))

    def zeropadding(self, img, nzpw , nzph):
        im = Image.fromarray(img)
        im = im.resize((self.w, self.h), Image.BILINEAR)
        im = np.asarray(im)
        hh = nzph + self.h
        ww = nzpw + self.w
        img_new = np.zeros((hh, ww))
        img_new[(hh-self.h)//2:(hh+self.h)//2, (ww-self.w)//2:(ww+self.w)//2] = im
        return img_new

    def resizeimg(self, wvl, img):
        """RGB 파장에 맞게 원본 이미지를 리사이징 + zero padding"""
        w_n = int(self.w * (self.wvl_B / wvl))
        h_n = int(self.h * (self.wvl_B / wvl))
        img_new = np.zeros((self.h*2, self.w*2))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(self.h*2 - h_n) // 2:(self.h*2 + h_n) // 2, (self.w*2 - w_n) // 2:(self.w*2 + w_n) // 2] = im
        return img_new


    def Fresnel(self, color):
        if color == 'green':
            wvl = self.wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = self.wvl_B
            image = self.img_B
        else:
            wvl = self.wvl_R
            image = self.img_R
        # resize image
        image = np.flip(image, axis=0)
        ps = self.scale / (2*self.w)
        phase = np.random.random((2*self.h, 2*self.w)) * 2 * np.pi  # random phase
        ph = np.exp(1j*phase)
        ph *= image
        ch2 = ph * h_frsn(ps, ps, self.w * 2, self.h * 2, self.zz, wvl)
        CH1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ch2)))
        result = CH1 * h_frsn(self.pp, self.pp, self.w * 2, self.h * 2, self.zz, wvl)
        result = result[self.h // 2: (3*self.h) // 2, self.w // 2: (3*self.w) // 2]
        result *= refwave(wvl, self.w, self.h, self.zz, self.pp, self.thetaX, self.thetaY)
        return result

    def ASM(self, color):
        if color == 'green':
            wvl = self.wvl_G
            image = self.img_g
        elif color == 'blue':
            wvl = self.wvl_B
            image = self.img_b
        else:
            wvl = self.wvl_R
            image = self.img_r
        phase = np.random.random((3 * self.h, 3 * self.w)) * 2 * np.pi  # random phase
        ph = np.exp(1j * phase)
        ph *= image
        CH = self.fft(ph)
        CH = CH * asm_kernel(wvl, self.zz, self.w, self.h, self.pp)
        result = self.ifft(CH)
        result = result[self.h: (2 * self.h), self.w: (2 * self.w)]
        result *= refwave(wvl, self.w, self.h, self.zz, self.pp, self.thetaX, self.thetaY)
        return result


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
