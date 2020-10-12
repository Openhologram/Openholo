import ophpy as oph


def test(method):
    if method == 'PointCloud-RS':
        # PointCloud base RS integral
        from ophpy import PointCloud
        plyfiles = PointCloud.get_pointcloud('PointCloud_Dice_RGB.ply')     # Read plyfiles
        red = PointCloud.RSIntegral(plyfiles, 'red')                        # Get complex hologram at red
        oph.getRGBImage(red, PointCloud.RSIntegral(plyfiles, 'green'), PointCloud.RSIntegral(plyfiles, 'blue'), 'test file-RS.bmp')

    elif method == 'PointCloud-Fresnel':
        # PointCloud base Fresnel integral
        from ophpy import PointCloud
        plyfiles = PointCloud.get_pointcloud('PointCloud_Dice_RGB.ply')     # Read plyfies
        red = PointCloud.FresnelIntegral(plyfiles, 'red')                   # Get complex hologram at Red color
        oph.getRGBImage(red, PointCloud.FresnelIntegral(plyfiles, 'green'), PointCloud.FresnelIntegral(plyfiles, 'blue'), 'test file-Frsn integral.bmp')

    elif method == '2D-ASM':
        # 2D image base CGH generation using ASM
        from ophpy import Image2D
        input_img = 'Dice_RGB.bmp'
        f = Image2D.Propagation(input_img)
        Red_image = f.ASM('red')  # Fresnel propagation using Single FFT
        f.getRGBImage(Red_image, f.ASM('green'), f.ASM('blue'), 'test file ASM.bmp', type='angle')

    elif method == '2D-Fresenl':
        # 2D image base CGH generation using Fresenl FFT
        from ophpy import Image2D
        input_img = 'Dice_RGB.bmp'
        f = Image2D.Propagation(input_img)
        Red_image = f.Fresnel('red')  # Fresnel propagation using Single FFT
        # plt.imshow(np.angle(Red_image))
        f.getRGBImage(Red_image, f.Fresnel('green'), f.Fresnel('blue'), 'test file Fresenl FFT.bmp', type='angle')

    elif method == 'Depthmap':
        # 3. Depthmap base CGH generation
        from ophpy import Depthmap
        input_img = 'Dice_RGB.bmp'
        input_depthmap = 'Dice_depth.png'
        D = Depthmap.Propagation(input_img, input_depthmap)
        Red_image = D.parallelCal('red')  # using parallel calculation
        D.getRGBImage(Red_image, D.parallelCal('green'), D.parallelCal('blue'), 'test file name.bmp', type='angle')

    else:
        print('Please enter "PointCloud-RS", "PointCloud-Fresnel", "2D-ASM", "2D-Fresnel", "Depthmap"  ')

if __name__ == '__main__':
    print('Please enter "PointCloud-RS", "PointCloud-Fresnel", "2D-ASM", "2D-Fresnel", "Depthmap"  ')
    inputs = input()
    test(inputs)



