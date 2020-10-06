import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840            # width
h = 2160            # height
pp = 3.6 * um       # SLM pixel pitch
nzp = w

scaleX = 0.03

@njit
def k(wvl):
    return (np.pi * 2) / wvl


@njit
def limits(u, z, wvl):
    # u is delta u
    return (1/wvl) * np.sqrt((2 * u * z)**2 + 1)


@njit
def asm_kernel(wvl, z):
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
                #(fx * fx + fy * fy) < (1 / (wvl * wvl)):
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


class FFT:
    def __init__(self, imgpath, f=1, angleX=0, angleY=0):
        self.zz = f
        self.imagein = np.asarray(Image.open(imgpath))
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.img_R = np.double(self.resizeimg(wvl_R, self.imagein[:, :, 0]))
        self.img_G = np.double(self.resizeimg(wvl_G, self.imagein[:, :, 1]))
        self.img_B = np.double(self.resizeimg(wvl_B, self.imagein[:, :, 2]))
        self.img_r = np.double(self.zeropadding(self.imagein[:,:,0]))
        self.img_g = np.double(self.zeropadding(self.imagein[:,:,1]))
        self.img_b = np.double(self.zeropadding(self.imagein[:,:,2]))

    def zeropadding(self, img, nzpw=2*w,nzph=2*h ):
        im = Image.fromarray(img)
        im = im.resize((w, h), Image.BILINEAR)
        im = np.asarray(im)
        hh = nzph + h
        ww = nzpw + w
        img_new = np.zeros((hh, ww))
        img_new[(hh-h)//2:(hh+h)//2, (ww-w)//2:(ww+w)//2] = im
        return img_new

    def resizeimg(self, wvl, img):
        """RGB 파장에 맞게 원본 이미지를 리사이징 + zero padding"""
        w_n = int(w * (wvl_B / wvl))
        h_n = int(h * (wvl_B / wvl))
        img_new = np.zeros((h*2, w*2))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(h*2 - h_n) // 2:(h*2 + h_n) // 2, (w*2 - w_n) // 2:(w*2 + w_n) // 2] = im
        return img_new

    def Cal(self, color):
        if color == 'green':
            wvl = wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_B
        else:
            wvl = wvl_R
            image = self.img_R
        # resize image
        zzz = 1 * self.zz
        ps = scaleX / (2*w)
        phase = np.random.random((2*h, 2*w)) * 2 * np.pi  # random phas
        ph = np.exp(1j*phase)
        ph *= image
        self.ch2 = ph * h_frsn(ps, ps, w + w, h + h, zzz, wvl)
        CH1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.ch2)))
        result = CH1 * h_frsn(pp, pp, w+w, h+h, zzz, wvl)
        result = result[h // 2: (3*h) // 2, w // 2: (3*w) // 2]
        result *= Refwave(wvl, zzz, self.thetaX, self.thetaY)
        return result

    def ASM(self, color):
        if color == 'green':
            wvl = wvl_G
            image = self.img_g
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_b
        else:
            wvl = wvl_R
            image = self.img_r
        phase = np.random.random((3 * h, 3 * w)) * 2 * np.pi  # random phas
        ph = np.exp(1j * phase)
        ph *= image
        CH = self.fft(ph)
        CH = CH * asm_kernel(wvl, self.zz)
        result = self.ifft(CH)
        result = result[h: (2 * h), w: (2 * w)]
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





if __name__ == '__main__':
    fs = FFT('imgin.bmp')
    fs.getRGBImage(fs.ASM('red'), fs.ASM('green'), fs.ASM('blue'), '200922 ASM test.bmp', type='angle')
    #fs.getRGBImage(fs.ASM('red'), fs.ASM('green'), fs.ASM('blue'), '200915 2D ASMFFT onaxis new4.bmp', type='angle')