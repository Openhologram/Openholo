from numba import njit, types, typed
import numpy as np
import matplotlib.pyplot as plt
import plyfile

# unit parameters
mm = 1e-3
um = mm * mm
nm = um * mm

@njit
def k(wvl):
    return (2 * np.pi) / wvl

def get_pointcloud(path):
    # converting point cloud data to numba type
    ans = typed.List()
    with open(path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        plydata = np.asarray(plydata.elements[1].data)

    for i in range(len(plydata)):
        raw = typed.List()
        [raw.append(types.float64(n)) for n in plydata[i]]
        ans.append(raw)
    return ans


@njit
def RSIntegral(plydata, color, propagation=1, angleX=0, angleY=0, Red=639 * nm, Green=525 * nm, Blue=463 * nm,
                    SLM_width=3840, SLM_height=2160, scaleXY=0.03, scaleZ=0.25, pixel_pitch=3.6 * um):
    """Fresnel approximation point propagation"""
    # define shift
    if color == 'green':
        num_color = 4
        wvl = Green
    elif color == 'blue':
        num_color = 5
        wvl = Blue
    else:
        wvl = Red
        num_color = 3
    shiftx = np.tan(angleX * (np.pi / 180)) * propagation
    shifty = np.tan(angleY * (np.pi / 180)) * propagation
    CHR = np.zeros((SLM_height, SLM_width))
    CHI = np.zeros((SLM_height, SLM_width))
    num = 0
    for n in plydata:
        num += 1
        ch_r = np.zeros((SLM_height, SLM_width))
        ch_i = np.zeros((SLM_height, SLM_width))
        x1 = (n[0] + shiftx) * scaleXY
        y1 = (n[1] + shifty) * scaleXY
        z1 = n[2] * scaleZ
        z = propagation - z1
        amp = n[num_color] * (z / wvl)
        for i in np.arange(SLM_height):
            for j in np.arange(SLM_width):
                x2 = (j - SLM_width / 2) * pixel_pitch
                y2 = -(i - SLM_height / 2) * pixel_pitch
                r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + z * z)
                t = (wvl * r) / (2 * pixel_pitch)
                if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
                    ch_r[i, j] = np.sin(k(wvl) * r) / r ** 2
                    ch_i[i, j] = np.cos(k(wvl) * r) / r ** 2
        CHR += ch_r * amp
        CHI += ch_i * amp
        print(num, 'th point', color, ' done')
    return (CHR + 1j * CHI)


@njit
def FresnelIntegral(plydata, color, propagation=1, angleX=0, angleY=0, Red=639 * nm, Green=525 * nm, Blue=463 * nm,
                    SLM_width=3840, SLM_height=2160, scaleXY=0.03, scaleZ=0.25, pixel_pitch=3.6 * um):
    """Fresnel approximation point propagation"""
    # define shift
    if color == 'green':
        num_color = 4
        wvl = Green
    elif color == 'blue':
        num_color = 5
        wvl = Blue
    else:
        wvl = Red
        num_color = 3
    shiftx = np.tan(angleX * (np.pi / 180)) * propagation
    shifty = np.tan(angleY * (np.pi / 180)) * propagation
    CHR = np.zeros((SLM_height, SLM_width))
    CHI = np.zeros((SLM_height, SLM_width))
    num = 0
    for n in plydata:
        num += 1
        ch_r = np.zeros((SLM_height, SLM_width))
        ch_i = np.zeros((SLM_height, SLM_width))
        x1 = (n[0] + shiftx) * scaleXY
        y1 = (n[1] + shifty) * scaleXY
        z1 = n[2] * scaleZ
        z = propagation - z1
        amp = n[num_color] * (z / wvl)
        for i in np.arange(SLM_height):
            for j in np.arange(SLM_width):
                x2 = (j - SLM_width / 2) * pixel_pitch
                y2 = -(i - SLM_height / 2) * pixel_pitch
                r = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) / (2 * z)
                t = (wvl * z) / (2 * pixel_pitch)
                if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
                    ch_r[i, j] = np.cos(k(wvl) * r)
                    ch_i[i, j] = np.sin(k(wvl) * r)
        CHR += ch_r * amp
        CHI += ch_i * amp
        print(num, 'th point done')
    return (CHR + 1j * CHI)


