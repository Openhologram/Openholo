# openholo_py

OpenHolo is an open source library which contains algorithms and their software implementation for generation of holograms to be applied in various fields. The goal behind the library development is facilitating production of digital holographic contents and expanding the area of their application. The developed by us open source library is a tool for computer generation of holograms, simulations and signal processing at various formats of 3D input data and properties of the 3D displays. Based on this, we want to lay the foundation for commercializing digital holographic service in various fields. The OpenHolo library has been developed in conjunction with the following development environment and provides Q & A, Open Source Software, Target Platform and BSD License.

For more information about Openholo, see the openholo homepage [Here](http://openholo.org/about)


## Installation

```python
python -m pip install ophpy
```

### Requirement
Numpy, Numba, Matplotlib, pillow, plyfile

### Procedure
Data input > Propagation > Encoding > Normalization > Save

## Usage

Generate Holographic Fringe pattern based on PointCloud data, 2D image, Depthmap image 


## Example
##### Point Cloud

```python
from ophpy import PointCloud

input_data = 'PointCloud_Dice_RGB.ply'
RS = PointCloud.Propagation(input_data, method='RS', angleY=1)

Red_image = RS.CalHolo('red')  # RS integral methods
plt.imshow(np.angle(Red_image))  # show phase angle data of red light fringe pattern

RS.getRGBImage(Red_image, RS.CalHolo('green'), RS.CalHolo('blue'), 'test file name.bmp', type='angle')
```

You can use Rayleigh-Sommerfeld diffraction or Fresnel diffraction integral

##### 2D image

```python
from ophpy import Image2D

input_img = 'Dice_RGB.bmp'
f = Image2D.Propagation(input_img, angleY=0.8)
Red_image = f.Fresnel('red')  # Fresnel propagation using Single FFT
plt.imshow(np.angle(Red_image))
f.getRGBImage(Red_image, f.Fresnel('green'), f.Fresnel('blue'), 'test file name.bmp', type='angle')
```

You can use Angular spectrum method and Fresnel propagation using Single FFT

##### Depthmap

```python
from ophpy import Depthmap

input_img = 'Dice_RGB.bmp'
input_depthmap = 'Dice_depth.bmp'
D = Depthmap.Propagation(input_img, input_depthmap)
Red_image = D.parallelCal('red')  # using parallel calculation
D.getRGBImage(Red_image, D.parallelCal('green'), D.parallelCal('blue'), 'test file name.bmp', tyep='angle')
plt.imshow(np.angle(Red_image))
```
