import matplotlib.pyplot as plt
import numpy as np
from ophpy import Depthmap, Image2D, PointCloud


def test(mode):
    if mode == 'PointCloud':
        # 1. Point Cloud base CGH generation
        input_data = 'PointCloud_Dice_RGB.ply'
        RS = PointCloud.Propagation(input_data, method='RS', angleY=1)
        Red_image = RS.CalHolo('red')  # RS integral methods
        plt.imshow(np.angle(Red_image))  # show phase angle data of red light fringe pattern
        RS.getRGBImage(Red_image, RS.CalHolo('green'), RS.CalHolo('blue'), 'test file name.bmp', type='angle')

    elif mode == '2Dimage':
        # 2. 2D image base CGH generation
        input_img = 'Dice_RGB.bmp'
        f = Image2D.Propagation(input_img, angleY=0.8)
        Red_image = f.Fresnel('red')  # Fresnel propagation using Single FFT
        plt.imshow(np.angle(Red_image))
        f.getRGBImage(Red_image, f.Fresnel('green'), f.Fresnel('blue'), 'test file name.bmp', type='angle')

    elif mode == 'Depthmap':
        # 3. Depthmap base CGH generation
        input_img = 'Dice_RGB.bmp'
        input_depthmap = 'Dice_depth.bmp'
        D = Depthmap.Propagation(input_img, input_depthmap)
        Red_image = D.parallelCal('red')  # using parallel calculation
        plt.imshow(np.angle(Red_image))
        D.getRGBImage(Red_image, D.parallelCal('green'), D.parallelCal('blue'), 'test file name.bmp', tyep='angle')
        


if __name__ == '__main__':
    test('PointCloud')  # enter type of source