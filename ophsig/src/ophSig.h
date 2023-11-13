/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/


#ifndef __ophSig_h
#define __ophSig_h

#include "tinyxml2.h"
#include "Openholo.h"
#include "sys.h"

#ifdef _WIN64
#ifdef SIG_EXPORT
#define SIG_DLL __declspec(dllexport)
#else
#define SIG_DLL __declspec(dllimport)
#endif
#else
#ifdef SIG_EXPORT
#define SIG_DLL __attribute__((visibility("default")))
#else
#define SIG_DLL
#endif
#endif

struct SIG_DLL ophSigConfig {
	int cols;
	int rows;
	Real_t width;
	Real_t height;
	Real_t NA;
	Real_t z;
	int wavelength_num;
	Real wavelength[3];
};

/**
* @addtogroup sig
//@{
* @details
This module is related method which works holographic core processing.

	* @section I. Holographic core processing

	1. Hologram convert method
		-   Convert to off-axis hologram
		-   Convert to horizontal parallax only hologram
		-   Convert to chromatic aberration compensated hologram

	2. Extraction of distance parameter method
		-   using sharpness function maximization method
		-   using axis transformation method

![](pics/ophsig/flowchart.png)

*/
//! @} sig

/**
* @addtogroup offaxis
//@{
* @detail
This module is related method which convert to off-axis hologram.

* @section Introduction
- Complex hologram records the amplitude and the complete phase of the diffracted optical field, reconstr
uction of the complex hologram can give the 3D image of an object without the twin-image and background
noise [1]. However, a conventional SLM cannot represent a complex hologram because the SLM can represent
either amplitude or phase. Because a conventional amplitude-only SLM can modulate the amplitude of an opti
cal field, we can represent either the real or the imaginary part of the complex hologram with DC bias.
However, when we reconstruct the 3D image of the object optically for 3D display using an amplitude-only SLM,
the reconstructed 3D image of the object is corrupted by the twin-image and background noise. On the other hand,
a phase-only SLM modulates only the phase of an optical field, and, thus, if we represent the phase of the comp
lex hologram using the phase-only SLM, the reconstructed 3D image is distorted by amplitude flattening.
- Convert complex holograms into off-axis holograms, which can reconstruct 3D images of objects without distor
tion due to twin image noise, background noise, and amplitude flattening [4].
![Figure 2. Concept of convet to off-axis hologram.](pics/ophsig/offaxis.png)

* @section Algorithm
-  In the off-axis hologram, the optical axis of the reference wave is tilted to that of the object wave.
The angle between the optical axes of the reference and object waves introduces a spatial carrier within
the hologram. The spatial carrier allows the separation of the desired 3D image of the object from the
twin-image noise and background noise in the reconstruction stage[2,3]. To convert the complex hologram to an
off-axis hologram, we multiply a spatial carrier term to complex hologram, \f$ H_{complex} \f$
\f[
H_{complex}^{spatial carrier}(x,y)=
\left | H_{complex} \right |
\exp \left ( j \angle H_{complex}\right )
\exp \left (j\frac{2\pi\sin\theta}{\lambda}x \right )
= \left | H \right |
\exp\left[j\left( \angle H_{complex}+\frac{2\pi\sin\theta}{\lambda}x\right ) \right] \qquad \left(1\right)
\f]
-  where \f$\angle\f$ represents the phase of a complex function, and \f$\theta\f$ is a tilted angle between the optical axes of
the reference and object waves. Note that we need to choose the off-axis angle, \f$\theta\f$, large enough to sepa
rate the desired 3D image from the twin-image and background noise[2,3]. The spatial carrier that is
added to the complex hologram separates the background noise as the zeroth-order beam and the twin image
as the first-order beam, which are spatially separated from the desired 3D image in the optical reconstr
uction[2,3]. To acquire an off-axis real hologram suitable for display on an amplitude-only SLM, we extract
the real part of Eq. (1) and add a DC bias to give
\f[
H_{complex}^{off-axis}(x,y)
=Re\left[H_{complex}^{spatialcarrier}(x,y)\right]+dc
=\left|H_{complex}^{spatialcarrier}(x,y)\right|\cos\left(\angle H_{complex}+\frac{2\pi\sin\theta}{\lambda} x\right)+dc \qquad \left(2\right)
\f]
-  where dc is a DC bias added to make the off-axis hologram become a positive value.
![Figure 2. Flowchart](@ref pics/ophsig/offaxis_flowchart.png)

* @section Reference
- [1] T.-C. Poon, T. Kim, G. Indebetouw, B. W. Schilling, M. H. Wu, K. Shinoda, and Y. Suzuki, “Twin-image elimination experiments for three-dimensional images in optical scanning holography,” Opt. Lett. 25, 215–217 (2000).
- [2] E. N. Leith and J. Upatnieks, “Reconstructed wavefronts and communication theory,” J. Opt. Soc. Am. 52, 1123–1130 (1962). 16.
- [3] E. N. Leith and J. Upatnieks, “Wavefront reconstruction with continuous-tone objects,” J. Opt. Soc. Am. 53, 1377–1381 (1963).
- [4] Y. S. Kim, T. Kim, T.-C. Poon, and J. T. Kim, “Three-dimensional display of a horizontal-parallax-only hologram,” Applied Optics Vol. 50, Issue 7, pp. B81-B87 (2011)

*/
//! @} offaxis

/**
* @addtogroup convHPO
//@{
* @detail
This module is related method which convert to horizontal parallax only hologram.

* @section Introduction
- The hologram contains all the information of a 3D object and require a large amount of transmission for use
in a holographic display. However, because the capacity of the information transmission channel is limited, a
great amount of research has been spent on holographic information reduction so as to facilitate, for example,
TV transmission of holograms. A horizontal-parallax-only (HPO) hologram has been proposed as an excellent way
to reduce the required amount of data for 3D display [1,2].
- Recently, HPO optical scanning holography(OSH) has been suggested as an electro-optical technique that actually
records the holographic information of a real object without vertical parallax [3]. However, the proposed HPO
OSH will introduce aberration upon optical reconstruction along the vertical direction if the vertical extent
of the asymmetrical Fresnel zone plate (FZP), which has been proposed to generate the HPO data from a real object,
is not small enough [3,4]. To eliminate aberration caused by the asymmetrical FZP, an algorithm that converts a
recorded full-parallax (FP) hologram to an HPO hologram was subsequently proposed [4]. The converted HPO hologram
was optically reconstruct using a conventional amplitude-only spatial light modulator (SLM) [5].

* @section Algorithm
- This method that converts a full-parallax hologram to an HPO hologram by using Gaussian low-pass filtering
and fringe-matched filtering. Although a full-parallax hologram of a 3D object can be considered as a collec
tion of 2D Fresnel zone plates (FZPs), an HPO hologram is a collection of 1D FZPs [3]. Figures 1(a) and 1(b) show
a 2D FZP and an asymmetrical FZP, respectively. The asymmetrical FZP shown in Fig. 1(b) illustrates an approx
imation to a line or 1D FZP by masking a slit along the x direction. Note that the asymmetrical FZP still has
curvature within its vertical extent if the slit size is not small enough, and hence it will generate aberrat
ion upon reconstruction of the hologram.
![Figure 1. (a) Full-parallax FZP, (b) asymmetrical FZP.](@ref pics/ophsig/hpo_fzp.png)
- Gaussian low-pass filtering along the vertical direction removes the high-frequency components of the object
along the vertical direction. This makes it possible to reduce the amount of data by sacrificing the vertical
parallax without losing the horizontal parallax. The filtered output becomes a hologram in which the object is
encoded by an asymmetrical FZP.
- Fringe-matched filter compensates the curvature of the Gaussian low-pass filtered hologram along the vertical
direction and gives an exact HPO hologram as an output [1,2]. This makes it possible to removes the curvature along
the vertical direction of the asymmetrical FZP.
- First, the full parallax complex hologram of the object obtained using the OSH setup is given by the following [6]:
\f[
H_{full}(x,y)=\int_{z_0-\Delta z}^{z_0+\Delta z}I_0(x,y,z) \otimes \frac{j}{\lambda z}
\times \exp \left \{ \left(\frac{\pi}{NA^2z^2}+j \frac{\pi}{\lambda z}\right)\left(x^2+y^2\right) \right \}dz \qquad \left(1\right)
\f]
- Where \f$NA\f$ represents the numerical aperture defined as the sine of the half-cone angle subtended by the TD-FZP,
\f$\lambda\f$ is the wavelength of the laser, \f$z_0\f$ is the depth location of the object, \f$2\Delta z\f$ is the depth range
of the object, and the symbol \f$\otimes\f$ denotes 2D convolution. The spectrum of the hologram is given by
\f[
H_{full}(k_x,k_y)=F\left\{H_{full}(x,y)\right\}
=\int_{z_0-\Delta z}^{z_0+\Delta z}I_0(k_x,k_y,z)
\times \exp \left \{ \left[-\frac{1}{4\pi}\left(\frac{\lambda}{NA}\right)+j \frac{\lambda z}{\lambda \pi}\right]
\left(k_x^2+k_y^2\right) \right \}dz \qquad \left(2\right)
\f]
- Where \f$ F\left\{.\right\}\f$. represents Fourier transformation with \f$\left(k_x,k_y\right)\f$ denoting spatial frequencies.
- Next apply a Gaussian low-pass filter along the vertical direction,
\f$ G_{low-pass}\left(k_x,k_y\right)=\exp\left[\frac{-1}{4\pi} \left(\frac{\lambda}{NA_g} \right)^2k_y^2 \right]\f$,
to the full-parallax hologram’s spectrum given by Eq. (2), where \f$ NA_g \f$ is a parameter that determines the cut
off frequency of the Gaussian low-pass filter. The filtered spectrum is then given by
\f[
H_{asym\ FZP}\left(k_x,k_y\right)
=H_{full}\left(x,y\right)G_{low-pass}\left(k_x,k_y\right)
=\int_{z_0-\Delta z}^{z_0+\Delta z}{I_0(k_x,k_y,z)
\times \exp\left\{\left[-\frac{1}{4\pi}\left(\frac{\lambda}{NA}\right)^2+j\frac{\lambda z}{4\pi}\right]k_x^2+
\left[-\frac{1}{4\pi}\left(\frac{\lambda}{{NA}_{lp}}\right)^2+j\frac{\lambda z}{4\pi}\right]k_y^2\right\}}dz \qquad \left(3\right)
\f]
- Where \f${\rm NA}_{lp}={\rm NA}_gNA/\sqrt{{\rm NA}^2+{NA}_g^2}\f$ is the NA of the FZP along the vertical direction.
Note that the Gaussian low-pass filtered hologram is a hologram in which the object’s cross-sectional images are encoded
with the asymmetrical FZP. As discussed earlier, the asymmetric FZP has curvature along the vertical direction. To remove
the curvature, use a fringe-matched filter, \f$F_{fm}\left(k_x,k_y\right)=exp\left[-j\lambda z_0/4\pi k_y^2\right]\f$,
that compensates the curvature along the vertical direction, where \f$z_0\f$ is the depth location of the object. Hence the fringe-adjusted filtered output becomes
\f[
H_{HPO}\left(k_x,k_y\right)
=H_{asym\ FZP}\left(k_x,k_y\right)F_{fm}\left(k_x,k_y\right)
=\int_{z_0-\Delta z}^{z_0+\Delta z}{I_0(k_x,k_y,z)\times \exp\left\{\left[-\frac{1}{4\pi}
\left(\frac{\lambda}{NA}\right)^2+j\frac{\lambda z}{4\pi}\right]k_x^2+{\left[-\frac{1}{4\pi}
\left(\frac{\lambda}{{NA}_{lp}}\right)^2+j\frac{\lambda\left(z-z_0\right)}{4\pi}\right]k}_y^2\right\}}dz \qquad \left(4\right)
\f]
- The HPO hologram in space domain is given by \f$ H_{HPO}\left(x,y\right)=F^{-1}\left\{H_{HPO}\left(k_x,k_y\right)\right\}\f$,
where \f$ F^{-1}\left\{.\right\} \f$. Represents the inverse Fourier transformation. Now, in the case that the depth range of
the object \f$ \left(2\Delta z\right)\f$ is smaller than the in-focus range of the line FZP along the vertical direction
\f$ \left(2\Delta z_{ver_dir}=2\lambda/\left({NA}_{lp}^2\right)\right)\f$, i.e., \f$ \Delta z\le\Delta z_{ver_dir}\f$, which is usually
true when we synthesize an HPO hologram for 3D display, \f$ z\approx z_0\f$ within the range of the object depth along the y direction,
and hence the last term of the exponential function become zero, i.e., \f$ \lambda\left(z-z_0\right)/4\pi\approx 0\f$. Equation (4) then becomes
\f[
H_{HPO}\left(k_x,k_y\right)
=\int_{z_0-\Delta z}^{z_0+\Delta z}{I_0(k_x,k_y,z)
\times \exp\left\{\left[-\frac{1}{4\pi}\left(\frac{\lambda}{NA}\right)^2+j\frac{\lambda z}{4\pi}\right]k_x^2
+{\left[-\frac{1}{4\pi}\left(\frac{\lambda}{NA_{lp}}\right)^2\right]k}_y^2\right\}}dz \qquad \left(5 \right )
\f]
- and its spatial domain expression is
\f[
H_{HPO}\left(x,y\right)=F^{-1}\left\{H_{HPO}\left(k_x,k_y\right)\right\}
=\int_{z_0-\Delta z}^{z_0+\Delta z}{I_0(x,y,z)\otimes\frac{j}{\lambda z}
\times \exp\left\{-\left[-\left(\frac{\pi}{NA^2z^2}\right)+j\frac{\pi}{\lambda z}\right]x^2
+\frac{\lambda}{NA_{lp}^2z^2}y^2\right\}}dz \qquad (6)
\f]
![Figure 2. Flowchart](@ref pics/ophsig/hpo_flowchart.png)
* @section Reference
- [1] P. St. Hilaire, S. A. Benton, and M. Lucente, “Synthetic aperture holography: a novel approach to three-dimensional displays,” J. Opt. Soc. Am. A 9, 11, 1969-1977 (1992).
- [2] H. Yoshikawa and H. Taniguchi, “Computer Generated Rainbow Hologram,” Opt. Rev. 6, 118 (1999).
- [3] T.-C. Poon, T. Akin, G. Indebetouw, and T. Kim, “Horizontal parallax-only electronic holography,” Opt. Express 13, 2427–2432 (2005).
- [4] T. Kim, Y. S. Kim, W. S. Kim, and T.-C. Poon, “Algorithm for converting full-parallax holograms to horizontal parallax-only holograms,” Opt. Lett. 34, 1231–1233 (2009).
- [5] Y. S. Kim, T. Kim, T.-C. Poon, and J. T. Kim, “Three-dimensional display of a horizontal-parallax-only hologram,” Applied Optics Vol. 50, Issue 7, pp. B81-B87 (2011)
- [6] T.-C. Poon, T. Kim, G. Indebetouw, M. H. Wu, K. Shinoda, and Y. Suzuki, “Twin-image elimination experiments for three-dimensional images in optical scanning holography,” Opt. Lett. 25, 215 (2000).

*/
//! @} convHPO

/**
* @addtogroup convCAC
//@{
* @detail
This module is related method which compensate of full color hologram with chromatic aberration.

* @section Introduction
- Recording holographic information of a real object as a form of electric signal has a long history [1,2].
Optical scanning holography (OSH) proposed to record a hologram of a real object using heterodyne scanning [3,4].
Twin-image noise in OSH was eliminated by recording a complex hologram using in-phase and quadrature(Q)-phase
heterodyne detection scheme [5]. speckle-free recording of a complex hologram using OSH has been demonstrated [6].
Recently full-color OSH has been proposed and shows that the full-color complex hologram of a real object can be
recorded by two dimensional (2D) scanning [7].
- Meanwhile, digital holography has been intensively investigated for recording a hologram of a real object for a
three-dimensional (3D) imaging system as well as industrial metrology applications [8,9]. Recently, color digital
holography has been proposed and chromatic aberration compensation techniques are investigated [11-15]. As in color
digital holography, the chromatic aberration issue has also been emerged in full-color OSH. In this paper, I will
investigate the chromatic aberration issue of full-color OSH and propose a digital filtering technique that compensates
the chromatic aberration.

* @section Algorithm
- The complex RGB holograms that are encoded pattern between RGB FZPs and object’s RGB reflectances are given by [7]:
\f[
H_n (x,y)=\int I_n (x,y,z)\otimes \frac{j}{\lambda_n z_n}\exp{\left(-\frac{\pi}{NA_n^2 z_n^2}+j\frac{\pi}{\lambda_n z_n} \right)(x^2+y^2 )}dz, \quad n={R,G,B} \qquad (1)
\f]
- Where \f$ \otimes \f$ represents 2D convolution operation, \f$ I_n(x,y,z) \f$ are the RGB reflectances of the object,
\f$ λ_n \f$ are the wave length of RGB beams, \f$ z_n \f$ are the distance between the focal points of RGB spherical waves
to the object’s RGB reflectances distributions and \f$ NA_n \f$ represent the numerical apertures defined as the sine of
the half-cone angle subtended by RGB spherical waves.
- Full-color OSH has chromatic aberration issue. fig. 1. represents the RGB spherical waves generated by L1 where
L1 is made of conventional material whose refractive index decreases as wavelength increases in visible range.
Since the refractive index of the material depends on the wavelength, the location of the focal points of the RGB
spherical waves are different as shown in fig. 1.
![Figure 1. RGB spherical waves generated by the lens L1.](@ref pics/ophsig/cac.png)
- The focal length of a conventional bi-convex lens is given by :
\f[
f=\frac{1}{(n_l-1)\left[\frac{2}{R}-\frac{(n_l-1)d}{n_l R^2}\right]} \qquad (2)
\f]
- Where \f$ n_l\f$ is the refractive index of the lens L1, \f$ R\f$ is the front and back radii of curvature and
\f$d\f$ is the center thickness of the lens L1. As shown in Eq (1), the RGB holograms are encoded patterns between
the spatial distribution of the object’s RGB reflectances and RGB FZPs. We note that the focal lengths of the RGB spheri
cal waves are different according the Eq (2). This makes the distances \f$(Z_n)\f$ from the focal points of the RGB spheri
cal waves to the object’s RGB reflectances distributions, and the numerical apertures \f$(NA_n=\sin\theta_n)\f$ of the RGB
FZPs depend on the wavelengths of RGB beams. Since the \f$Z_n\f$ and the \f$NA_n\f$ of the RGB FZPs are different according
to the RGB beams, the recorded RGB holograms are reconstructed at different depth locations with different divergence angles.
This causes chromatic aberration as in the conventional imaging system.
- We need to match the NAs and focal lengths of RGB FZPs each other. Because focal length and NAs of the RGB FZPs in recorded
RGB holograms are different according to the wavelengths. Chromatic aberration compensation filter (CAC) matches the NAs and
focal lengths of the RGB FZPs of the recorded RGB hologram. The fig. 2 shows the CAC filter that matches the NAs and focal
lengths of the GB FZPs in the GB holograms to those of the R FZP in the R hologram, where \f$(k_x,k_y)\f$ is the spatial frequencies,
\f$G_n\f$ are the extent of the CAC filter and \f$z_n\f$ are the focal length differences as shown in fig. 1.
- The extent of CAC filter \f$G_n\f$ match the NAs of the GB FZPs to the NA of the R hologram. The fringes of the CAC filter
determined by \f$\lambda_n z_n\f$ match the focal lengths of GB FZPs to the focal length of the R FZP.
![Figure 2. Flowchart.](@ref pics/ophsig/cac_flowchart.png)

* @section Reference
- [1] C. Burckhardt and L. Enloe, "Television transmission of holograms with reduced resolution requirements on the camera tube," Bell Syst. Tech.Jour 45, 1529–1535 (1969).
- [2] J. Berrang, "Television transmission of holograms using a narrow-bandvideo signal," Bell Syst. Tech. J 879–887 (1970).
- [3] T. C. Poon and a Korpel, "Optical transfer function of an acousto-opticheterodyning image processor," Opt. Lett. 4, 317–9 (1979).
- [4] T.-C. Poon, "Scanning holography and two-dimensional image processing by acousto-optic two-pupil synthesis," J. Opt. Soc. Am. A 2, 521 (1985).
- [5] T. C. Poon, T. Kim, G. Indebetouw, B. W. Schilling, M. H. Wu, K.Shinoda, and Y. Suzuki, "Twin-image elimination experiments for threedimensional images in optical scanning holography," Opt. Lett. 25, 215–7 (2000).
- [6] Y. S. Kim, T. Kim, S. S. Woo, H. Kang, T. C. Poon, and C. Zhou,"Speckle-free digital holographic recording of a diffusely reflecting object," Opt. Express 21, 8183–8189 (2013).
- [7] H. Kim, Y. S. Kim, T. Kim, “Full-color optical scanning holography with common Red, Green and Blue channels,” Appl. Opt. 50, B81–B87 (2016).
- [8] U. Schnars, “Direct phase determination in hologram interferometry with use of digitally recorded holograms,” J. Opt. Soc. Am. A 11, 2011–2015 (1994).
- [9] I. Yamaguchi and T. Zhang, “Phase-shifting digital holography,” Opt.Lett. 22, 1268–1270 (1997).
- [10] U. Schanrs and W. Jueptner, Digital Holography, Springer (2005).
- [11] I. Yamaguchi, T. Matsumura, and J. Kato, “Phase-shifting color digital holography,” Opt. Lett. 27, 1108–1110 (2002).
- [12] P. Memmolo, A. Finizio, M. Paturzo, P. Ferraro, and B. Javidi “Multiwavelengths digital holography: reconstruction, synthesis and display of holograms using adaptive transformation” Opt. Lett. 37, No. 9, 1445-1447 (2012)
- [13] M. K. Kim “Full color natural light holographic camera” Opt. Express 21, No. 8, 9636-9642 (2013)
- [14] T. Tahara, Y. Ito, Y. Lee, P. Xia, J. Inoue, Y. Awatsuji, K. Nishio, S. Ura, T. Kubota, and O. Matoba “Multiwavelength parallel phase-shifting digital holography using angular multiplexing” Opt. Lett. 38, No. 15, 2789-2791 (2013)
- [15] J. Dohet-Eraly, C. Yourassowsky, and F. Dubois “Refocusing based on amplitude analysis in color digital holographic microscopy” Opt. Lett. 39, No. 5, 1109-1112 (2014)

*/
//! @} convCAC

/**
* @addtogroup getAT
//@{
* @detail
This module is related method which extraction of distance parameter using axis transformation method.

* @section Introduction
- OSH is reconstructed digitally by convolving the complex conjugate of Fresnel zone plate (FZP) with the hologram.
This is the same as conventional digital holography, in which the FZP’s distance parameter is set according to the
depth location of the objects. However, since the depth location of the objects is unknown, digital reconstructions
with different distance parameters are required until we get a focused image.
- Unfortunately, this searching process is time consuming and requires a manual process. Recently several numerical
techniques that extract the distance parameter directly from the hologram without reconstructions have been proposed [1-3];
however, these involve a search algorithm [2,3] or a tracking process [1]. Most recently, an auto-focusing technique based
on the Wigner distribution analysis of a hologram has been proposed [4].
- In the proposed technique, we extract the distance parameter directly from the hologram without any searching or tracking
process. However, a manual process that measures the slope of the Wigner distribution output is required in order to determine
the distance parameter. Therefore, we propose to extract the distance parameter directly from the hologram using axis
transformation without any manual processes.

* @section Algorithm
- First, the complex hologram is filtered by a Gaussian low-pass filter with transfer function. The Gaussian low-pass filtered
hologram is given by the following equation:
\f[
H_{com}^{lp}\left(x,y\right)=F^{-1}\left\{F\left[H_{com}\left(x,y\right)\right]\times G\left(k_x,k_y\right)\right\}
=\int_{z_0-\frac{1}{2}\delta z}^{z_0+\frac{1}{2}\delta z}I_o\left(x,y,z \right)\otimes\frac{j}{\lambda z}
\exp\left[\left(\frac{-\pi}{a_{lp} z^2}-j\frac{\pi}{\lambda z}\right)\left(x^2+y^2\right)\right]dz \qquad (1)
G\left(k_x,k_y\right)=exp\left\{-\pi\left[\frac{\lambda}{2\pi{NA}_g}\right]^2\left({k_x}^2+{k_y}^2\right)\right\}
\f]
- Where \f$F{.}\f$ represents the Fourier transform operator, \f$a_{lp}\left(z\right)=NA_gNA/\sqrt{NA^2+NA_g^2}\times z\f$
determines the radius of the Gaussian low pass filtered hologram. Hence, the Reyleigh range of the Gaussian low-pass filtered
hologram evolves into the following equation:
\f[
\Delta z=2{\lambda z}^2/{{(a}_{lp}z}^2\ \pi)=2\lambda/\pi\times({NA}^2+{NA}_g^2)/{({NA}_gNA)}^2
\f]
- In Gaussian low pass filtering, we set \f$NA_g\f$ such that the Rayleigh range of the FZP is larger than the depth range of
the object, i.e. \f$\Delta z\geq\delta z\f$. The radius of the scanning beam pattern is approximately constant within the depth
range of the object. This makes the phase of the Gaussian low pass filtered hologram stationary within the depth range of the
specimen [4, 5]; thus, we can extract the FZP which contains the information only about the distance parameter from the hologram.
- Second, a real-only spectrum hologram in the frequency domain is synthesized and is given by the following:
\f[
H_{r-only}^{lp}\left(k_x,k_y\right)=Re\left\{F\left[Re\left(H_{com}^{lp}\left(x,y\right)\right)\right]\right\}
+jRe\left\{F\left[Im\left(H_{com}^{lp}\left(x,y\right)\right)\right]\right\} \qquad (2)
\f]
- After projecting the real-only spectrum hologram onto the ky direction, we filter the square of the projected real-only spectrum
hologram by a power-fringe-adjusted filter [6-9]. The power-fringe-adjusted filtered output is given by:
\f[
H_{FZP}\left(k_x\right)=\frac{\left[\int{H_{r-only}^{lp}\left(k_x,k_y\right)dk_y}\right]^2}{\left|\int{H_{r-only}^{lp}
\left(k_x,k_y\right)dk_y}\right|^2+\delta}\approx exp\left(j\frac{\lambda z_0}{2\pi}k_x^2\right) \qquad (3)
\f]
- Eq. (3), the filtered output is a chirping signal whose chirping rate is determined by the distance parameter. Note that the
filtered output is the one-dimensional FZP with the distance parameter.
- Third, the real part of Eq. (3) is extracted and is given by the following:\ .\
\f[
H_{FZP}^{Re}\left(k_x\right)=Re\left\lfloor H_{FZP}\left(k_x\right)\right\rfloor \approx cos\left(\frac{\lambda z_0}{2\pi}k_x^2\right) \qquad (4)
\f]
- \f$H_{FZP}^{Re}\left(k_x\right)\f$ shown as fig.1-a.
- Fourth, the transformation from original frequency axis to new-frequency axis using interpolation.
\f[
k_x^{new}=k_x^2\ , H_{FZP}^{Re}\left(k_x^{new}\right)\approx cos\left(\frac{\lambda z_0}{2\pi}k_x^{new}\right) \qquad (5)
\f]
- Note that this sinusoidal signal has a single frequency and the frequency of the signal is directly proportional to the distance parameter.
Hence, the inverse Fourier transformation of Eq.(5) expresses the delta function pair in the new spatial axis:
\f[
h_{FZP}^{Re}\left(x^{new}\right)
=\mathbf{F}^{-\mathbf{1}}\left\{H_{FZP}^{Re}\left(k_x^{new}\right)\right\}
\approx\frac{1}{2}\delta\left(x^{new}-\frac{\lambda z_0}{2\pi}\right)+\frac{1}{2}\delta\left(x^{new}+\frac{\lambda z_0}{2\pi}\right) \qquad (6)
\f]
- Note that the location of the delta function pair gives the distance parameter. This can be extracted directly by detecting the location of the maximum value of Eq. (6).
![Figure 1. Flowchart.](@ref pics/ophsig/at_flowchart.png)

* @section Reference
- [1] P. Ferraro, G. Coppola, S. D. Nicola, A. Finizio, and G. Pierattini, “Digital holographic microscope with automatic focus tracking by detecting sample displacement in real time,” Opt. Lett. 28, 1257-1259 (2003).
- [2] M. Liebling and M. Unser, “Autofocus for digital Fresnel holograms by use of a Fresnelet-sparsity criterion,” J. Opt. Soc. Am. A 21, 2424-2430 (2004).
- [3] P. Langehanenberg, B. Kemper, D. Dirksen, and G. von Bally, “Autofocusing in digital holographic phase contrast microscopy on pure phase objects for live cell imaging,” Appl. Opt. 47, D176 (2008).
- [4] T. Kim and T.-C. Poon, “Auto-focusing in optical scanning holography,” Appl. Opt. 48, H153-H159 (2009).
- [5] T. Kim, Y. S. Kim, W. S. Kim, and T.-C. Poon, “Algorithm for converting full-parallax holograms to horizontal parallax -only holograms,” Opt. Lett. 34, 1231-1233 (2009).
- [6] T. Kim and T.-C. Poon, “Extraction of 3-D location of matched 3-D object using power fringe-adjusted filtering and Wigner analysis,” Opt. Eng. 38, 2176-2183 (1999).
- [7] T. Kim and T.-C. Poon, “Experiments of depth detection and image recovery of a remote target using a complex hologram,” Opt. Eng. 43, 1851-1855 (2004).
- [8] T. Kim, T.-C. Poon, and G. Indebetouw, ‘‘Depth detection and image recovery in remote sensing by optical scanning holography,’’ Opt. Eng. 41, 1331-1338 (2002).
- [9] P. Klysubun, G. Indebetouw, T. Kim, and T.-C. Poon, ‘‘Accuracy of three-dimensional remote target location using scanning holographic correlation,’’ Opt. Comm. 184, 357 -366 (2000).
*/
//! @} getAT

/**
* @addtogroup getSF
//@{
* @detail
This module is related method which extraction of distance parameter using sharpness function maximization method.

* @section Introduction
- We use autofocusing to capture in-focus images. It is based on sharpness of images and various of autofocusing algorithms
have been proposed. It represents a peak when the image is in-focus and drops when the image goes out-of-focus. It can relate
to holography signal process. Hologram has a depth information of object and is reconstructed at that point. The sharpness
of the reconstructed hologram image changes with the change of the depth position.
- If the depth of focus is not correct, the reconstructed hologram can not have a clear image. It means the same as in-focus
image phenomenon. For this reasons, we will discuss the hologram signal processing using the sharpness functions.

* @section Algorithm
- Brenner function [1] : A focus function f(Z) is calculated which is a measure of the average change in gray level between
pairs of points separated by n pixels. f(Z) is a maximum when the image is in focus. and is given by
\f[
f(Z)=\sum_{j}\sum_{i}\left| G_i(Z)-G_{ij}(Z)\right|^2
\f]
- Where the index (i) ranges over all image points, in order along a scan line (j); n is a small integer; Z is the Z-axis,
or focus position; and \f$G_i\f$ is the transmission gray level for point i. A value of n equal to 2 gives a good signal to noise ratio.
![Figure 1. Concept of searching distance parameter.](@ref pics/ophsig/sf_concept.png)
- Reconstruct the hologram to sequential depth positions using Fresnel diffraction method. then we can obtain \f$ f(Z) \f$ of reconstructed hologram image.
if \f$f(Z)\f$ is maximum value, value of \f$Z\f$ is distance parameter of hologram.

* @section Reference
- [1] J. Brenner et al., "An Automated Microscope for Cytologic Research - A Preliminary Evaluation", Journal of Histochemistry and Cytochemistry, vol. 24, no. 1, pp. 100-111, 1976

*/
//! @} getSF


/**
* @addtogroup PSDH
//@{
* @details

* @section Algorithm
- Extract complex field from 4 interference patterns with 90 degree phase shifts of the reference wave
Store the result complex field to the member variable ComplexH
![](pics/ophsig/psdh/psdh_concept.png)
- Extract complex field from 3 interference patterns with arbitrary phase shifts of the reference wave
Store the result complex field to the member variable ComplexH
* @section Reference
[ref] L.Z. CAi, Q. Liu, and X.L. Yang, Opt. Lett. 28(19) 1808 (2003)

*/
//! @} PSDH

/**
* @ingroup sig
* @brief
* @author
*/
class SIG_DLL ophSig : public Openholo
{
protected:

	virtual ~ophSig(void) = default;

	virtual void ophFree(void);
	bool is_CPU;
	ophSigConfig _cfgSig;
	OphComplexField* ComplexH;
	fftw_plan bwd_plan, fwd_plan;

	int _wavelength_num;
	Real_t _radius;
	Real_t _foc[3];



	/**
	* @ingroup offaxis
	* @brief			Function for Convert complex hologram to off-axis hologram by using CPU
	* @param angleX		X-axis angle of off-axis
	* @param angleY		Y-axis angle of off-axis
	* @return if works well return 0  or error occurs return -1
	*/
	bool sigConvertOffaxis_CPU(Real angleX, Real angleY);
	/**
	* @ingroup offaxis
	* @brief			Function for Convert complex hologram to off-axis hologram by using GPU
	* @param angleX		X-axis angle of off-axis
	* @param angleY		Y-axis angle of off-axis
	* @return if works well return 0  or error occurs return -1
	*/
	bool sigConvertOffaxis_GPU(Real angleX, Real angleY);
	/**
	* @ingroup convHPO
	* @brief			Function for convert complex hologram to horizontal parallax only hologram by using CPU
	* @param depth		Position from hologram plane to propagation hologram plane
	* @param redRate	data reduction rate
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertHPO_CPU(Real depth, Real_t redRate);
	/**
	* @ingroup convHPO
	* @brief			Function for convert complex hologram to horizontal parallax only hologram by using GPU
	* @param depth		Position from hologram plane to propagation hologram plane
	* @param redRate	data reduction rate
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertHPO_GPU(Real depth, Real_t redRate);

	/**
	* @ingroup convCAC
	* @brief			Function for Chromatic aberration compensation filter by using CPU
	* @detail
	* @param red		Red wavelength
	* @param green		Green wavelength
	* @param blue		Blue wavelength
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertCAC_CPU(double red, double green, double blue);
	/**
	* @ingroup convCAC
	* @brief			Function for Chromatic aberration compensation filter by using GPU
	* @param red		Red wavelength
	* @param green		Green wavelength
	* @param blue		Blue wavelength
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertCAC_GPU(double red, double green, double blue);
	/**
	* @ingroup getAT
	* @brief		Extraction of distance parameter using axis transfomation by using CPU
	* @return		Result distance
	*/
	double sigGetParamAT_CPU();
	/**
	* @ingroup getAT
	* @brief		Extraction of distance parameter using axis transfomation by using GPU
	* @return		Result distance
	*/
	double sigGetParamAT_GPU();
	/**
	* @ingroup getSF
	* @brief			Extraction of distance parameter using sharpness functions by using CPU
	* @param zMax		Maximum value of distance on z axis
	* @param zMin		Minimum value of distance on z axis
	* @param sampN		Count of search step
	* @param th			Threshold value
	* @return			Result distance
	*/
	double sigGetParamSF_CPU(float zMax, float zMin, int sampN, float th);
	/**
	* @ingroup getSF
	* @brief			Extraction of distance parameter using sharpness functions by using GPU
	* @param zMax		Maximum value of distance on z axis
	* @param zMin		Minimum value of distance on z axis
	* @param sampN		Count of search step
	* @param th			Threshold value
	* @return			Result distance
	*/
	double sigGetParamSF_GPU(float zMax, float zMin, int sampN, float th);

	/**
	* @brief			Function for propagation hologram by using CPU
	* @param depth		Position from hologram plane to propagation hologram plane
	* @return		    If works well return 0  or error occurs return -1
	*/
	bool propagationHolo_CPU(float depth);
	/**
	* @brief			Function for propagation hologram by using GPU
	* @param depth		Position from hologram plane to propagation hologram plane
	* @return		    If works well return 0  or error occurs return -1
	*/
	bool propagationHolo_GPU(float depth);

	bool Color_propagationHolo_GPU(float depth);
public:
	/**
	* @brief Constructor
	*/
	explicit ophSig(void);
	/**
	* @brief          Load bmp or bin file
	* @param real     Real data file name
	* @param imag     Imag data file name
	* @return         If works well return 0  or error occurs return -1
	*/
	bool load(const char *real, const char *imag);
	/**
	* @brief          Save data as bmp or bin file
	* @param real     Real data file name
	* @param imag     Imag data file name
	* @return         If works well return 0  or error occurs return -1
	*/
	bool save(const char *real, const char *imag);
	bool save(const char *real);

	/**
	* @brief          Load data as ohc file
	* @param fname    File name
	* @return         If works well return 0  or error occurs return -1
	*/
	bool loadAsOhc(const char *fname);
	/**
	* @brief          Save data as ohc file
	* @param fname    File name
	* @return         If works well return 0  or error occurs return -1
	*/
	bool saveAsOhc(const char *fname);
	/**
	* @brief          Linear interpolation
	* @param X		  Sample point
	* @param src      Sample values
	* @param Xq       Query points
	* @param Xq       Query values
	*/
	template<typename T>
	void linInterp(vector<T> &X, matrix<Complex<T>> &src, vector<T> &Xq, matrix<Complex<T>> &dst);
	/**
	* @brief           Generate linearly spaced vector
	* @param first     First number of vector
	* @param last      Last number of vector
	* @param len       Vector with specified number of values
	* @return          Result vector
	*/
	template<typename T>
	vector<T> linspace(T first, T last, int len) {
		vector<Real> result(len);
		Real step = (last - first) / (len - 1);
		for (int i = 0; i < len; i++) { result[i] = first + i * step; }
		return result;
	}
	/**
	* @brief         Function for extracts Complex absolute value
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void absMat(matrix<Complex<T>>& src, matrix<T>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst.mat[i][j] = sqrt(src.mat[i][j][_RE] * src.mat[i][j][_RE] + src.mat[i][j][_IM] * src.mat[i][j][_IM]);
			}
		}
	}
	/**
	* @brief         Function for extracts real absolute value
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void absMat(matrix<T>& src, matrix<T>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst.mat[i][j] = abs(src.mat[i][j]);
			}
		}
	}
	/**
	* @brief         Function for extracts Complex phase value
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void angleMat(matrix<Complex<T>>& src, matrix<T>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				angle(src(i, j), dst(i, j));
			}
		}
	}
	/**
	* @brief         Function for extracts Complex conjugate value
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void conjMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst(i, j) = src(i, j).conj();

			}
		}
	}
	/**
	* @brief         Function for extracts exponent e(x), where x is complex number
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void expMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst.mat[i][j][_RE] = exp(src.mat[i][j][_RE]) * cos(src.mat[i][j][_IM]);
				dst.mat[i][j][_IM] = exp(src.mat[i][j][_RE]) * sin(src.mat[i][j][_IM]);
			}
		}
	}
	/**
	* @brief         Function for extracts exponent e(x), where x is real number
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void expMat(matrix<T>& src, matrix<T>& dst) {
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst.mat[i][j] = exp(src.mat[i][j]);
			}
		}
	}
	/**
	* @brief         Function for extracts mean of matrix
	* @param src     Input data
	* @param dst     Output data
	*/
	template<typename T>
	void meanOfMat(matrix<T>& src, T &dst) {
		dst = 0;
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				dst += src(i, j);
			}
		}
		dst = dst / (src.size[_X] * src.size[_Y]);
	}
	/**
	* @brief         Function for extracts maximum of matrix , where matrix is real number
	* @param src     Input data
	* @return        Output data
	*/
	Real maxOfMat(matrix<Real>& src) {
		Real max = MIN_REAL;
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				if (src(i, j) > max) max = src(i, j);
			}
		}
		return max;
	}
	/**
	* @brief         Function for extracts maximum of matrix , where matrix is complex number
	* @param src     Input data
	* @return        Output data
	*/
	Complex<Real> maxOfMat(matrix<Complex<Real>>& src)
	{
		Real max = MIN_REAL;
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				if (src(i, j)[_RE] > max) max = src(i, j)[_RE];
				//if (src(i, j)[_IM] > max) max = src(i, j)[_IM];
			}
		}
		return max;
	}
	/**
	* @brief         Function for extracts minimum of matrix , where matrix is real number
	* @param src     Input signal
	*/
	//template<typename T>
	Real minOfMat(matrix<Real>& src) {
		Real min = MAX_REAL;
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				if (src(i, j) < min) min = src(i, j);
			}
		}
		return min;
	}
	/**
	* @brief         Function for extracts minimum of matrix , where matrix is complex number
	* @param src     Input data
	*/
	Complex<Real> minOfMat(matrix<Complex<Real>>& src) {
		Real min = MAX_REAL;
		for (int i = 0; i < src.size[_X]; i++)
		{
			for (int j = 0; j < src.size[_Y]; j++)
			{
				if (src(i, j)[_RE] < min) min = src(i, j)[_RE];
			}
		}
		return min;
	}

	/**
	* @brief         Shift zero-frequency component to center of spectrum
	* @param src     Input data
	* @param dst     Output data
	*/
	void fftShift(matrix<Complex<Real>> &src, matrix<Complex<Real>> &dst)
	{
		if (src.size != dst.size) {
			dst.resize(src.size[_X], src.size[_Y]);
		}
		int xshift = src.size[_X] / 2;
		int yshift = src.size[_Y] / 2;
		for (int i = 0; i < src.size[_X]; i++)
		{
			int ii = (i + xshift) % src.size[_X];
			for (int j = 0; j < src.size[_Y]; j++)
			{
				int jj = (j + yshift) % src.size[_Y];
				dst.mat[ii][jj][_RE] = src.mat[i][j].real();
				dst.mat[ii][jj][_IM] = src.mat[i][j].imag();
			}
		}
	}

	/**
	* @brief		Function for Fast Fourier transform 1D
	* @param src	Input data
	* @param dst	Output data
	* @param sign	sign = OPH_FORWARD is fft and sign= OPH_BACKWARD is inverse fft
	* @param flag	flag = OPH_ESTIMATE is fine best way to compute the transform but it is need some time, flag = OPH_ESTIMATE is probably sub-optimal
	*/
	template<typename T>
	void fft1(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* @brief		Function for Fast Fourier transform 2D
	* @param src	Input data
	* @param dst	Output data
	* @param sign	sign = OPH_FORWARD is fft and sign= OPH_BACKWARD is inverse fft
	* @param flag	flag = OPH_ESTIMATE is fine best way to compute the transform but it is need some time, flag = OPH_ESTIMATE is probably sub-optimal
	*/
	template<typename T>
	void fft2(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* @brief Function for Read parameter
	* @param fname file name
	* @return if works well return 0  or error occurs return -1
	*/
	bool readConfig(const char* fname);
	void Parameter_Set(int nx, int ny, double width, double height , double NA );
	void wavelength_Set(double wavelength);
	void focal_length_Set(double red , double green, double blue, double rad);
	void Data_output(uchar  *data, int pos ,int bitpixel);
	void Wavenumber_output(int &wavenumber);

	/**
	* @ingroup offaxis
	* @brief			Function for Convert complex hologram to off-axis hologram
	* @param angleX		X-axis angle of off-axis
	* @param angleY		Y-axis angle of off-axis
	* @return if works well return 0  or error occurs return -1
	*/
	bool sigConvertOffaxis(Real angleX, Real angleY);
	bool cvtOffaxis_CPU(Real angleX, Real angleY);
	void cvtOffaxis_GPU(Real angleX, Real angleY);


	/**
	* @ingroup convHPO
	* @brief			Function for convert complex hologram to horizontal parallax only hologram
	* @param depth		Position from hologram plane to propagation hologram plane
	* @param redRate	data reduction rate
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertHPO(Real depth, Real_t redRate);


	/**
	* @ingroup convCAC
	* @brief			Function for Chromatic aberration compensation filter
	* @param red		Red wavelength
	* @param Green		Green wavelength
	* @param Blue		Blue wavelength
	* @return			If works well return 0  or error occurs return -1
	*/
	bool sigConvertCAC(double red, double green, double blue);

	/**
	* @brief			Function for propagation hologram (class data)
	* @param depth		Position from hologram plane to propagation hologram plane
	* @return		    If works well return 0  or error occurs return -1
	*/
	bool propagationHolo(float depth);
	/**
	* @brief			Function for propagation hologram
	* @param complexH	Input data
	* @param depth		position from hologram plane to propagation hologram plane
	* @return			Output data
	*/
	OphComplexField propagationHolo(OphComplexField complexH, float depth);

	/**
	* @ingroup getAT
	* @brief		Extraction of distance parameter using axis transfomation
	* @return		Result distance
	*/
	double sigGetParamAT();

	/**
	* @ingroup getSF
	* @brief			Extraction of distance parameter using sharpness functions
	* @param zMax		Maximum value of distance on z axis
	* @param zMin		Minimum value of distance on z axis
	* @param sampN		Count of search step
	* @param th			Threshold value
	* @return			Result distance
	*/
	double sigGetParamSF(float zMax, float zMin, int sampN, float th);

	/**
	* @brief			Function for select device
	* @param is_CPU		If is_CPU  = true  using CPU , is_CPU  = false  using GPU
	*/
	void setMode(bool is_CPU);
	/**
	* @brief			Function for move data from matrix<Complex<Real>> to Complex<Real>
	* @param src		Input martix data
	* @param dst		Output data
	* @param nx			X_axis size
	* @param ny			Y_axis size
	*/
	void cField2Buffer(matrix<Complex<Real>>& src, Complex<Real> **dst, int nx, int ny);

	/**
	* @brief			Function for move Color data from matrix<Complex<Real>> to Complex<Real>
	* @param src		Input martix data
	* @param dst		Output data
	* @param nx			X_axis size
	* @param ny			Y_axis size
	*/
	void ColorField2Buffer(matrix<Complex<Real>>& src, Complex<Real> **dst, int nx, int ny);

	/**
	* @ingroup PSDH
	* @brief Extraction of complex field from 4 phase shifted interference patterns
	* @details
	* @param fname0, fname90, fname180, fname270 Input image files for 4 interference patterns
	* @return if works well return 0  or error occurs return -1
	*/
	bool getComplexHFromPSDH(const char* fname0, const char* fname90, const char* fname180, const char* fname270);

	/**
	* @ingroup PSDH
	* @brief Extraction of complex field from 3 phase shifted interference patterns with arbitrary unknown shifts
	* @details
	* @param f0, f1, f2 Input image files for 3 interference patterns
	* @param fOI, Input image file for object wave intensity 
	* @param nIter The number of the iterations in estimating the phase shift
	* @return if works well return 0  or error occurs return -1
	*/
	bool getComplexHFrom3ArbStepPSDH(const char* f0, const char* f1, const char* f2, const char* fOI, const char* fRI, int nIter);

};

#endif // !__ophSig_h

