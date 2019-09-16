#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:48:01 2019

     MATCHED FILTER code
     Adapted for usage by B. Casaponsa and D. Balbas
     Modified by D. Balbas
     Used with the author's permission

@author: herranz
"""

### --- IMPORTS AND DEFINITIONS _----------------------------------------------

import numpy as np
import numba
import warnings
import matplotlib.pyplot as plt

from   astropy.modeling  import models,fitting
from   skimage.feature   import peak_local_max
from   scipy             import ndimage
from   scipy.interpolate import interp1d

fwhm2sigma   = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))
sigma2fwhm   = 1.0/fwhm2sigma

### --- MAIN PROGRAM ----------------------------------------------------------

def matched_filter(input_image,            # input image (2D array)
                   fwhm_pix  = 2.0,        # Gaussian FWHW (pixel units)
                   bins      = 'auto',     # number of bins for power spectrum estimation. It can be 'auto' or an integer
                   iterative = True,       # if True, the routine tries to first detect/remove bright sources
                   threshold = 5.0):       # threshold to be appikled if iterative == True
    """
     This routine assumes that the profile of the sources is Gaussian and that
     the width of the sources is known. It only works in that case. The
     routine makes a first filtering of the data, identifies peaks in
     the filtered_image image with SNR greater than threshold and removes them from the
     image by consecutive fits to the data in small patches
     around the peaks, in order to get a clean "background" with the
     which re-calculate the matched filter.

    """

    sigma_pix      = fwhm2sigma * fwhm_pix
    s              = input_image.shape
    size0          = s[0]
    if bins == 'auto':
        nbins = size0//4
    else:
        nbins = bins

    # Pad with zeros a region around the image, to avoid strong border effects

    l              = 2*int(fwhm_pix)
    padscheme      = ((l,l),(l,l))
    padded_image   = np.pad(input_image.copy(),
                            padscheme,
                            mode='reflect')

    s              = padded_image.shape
    size           = s[0]

#   Gaussian profile (in real and Fourier space):

    gauss_beam     = Gaussian_profile(size,fwhm_pix=fwhm_pix)
    gauss_beam_f   = np.abs(np.fft.fftshift(np.fft.fft2(gauss_beam)))

#   FFT and power spectrum map of the original data:

    padded_image_f = np.fft.fftshift(np.fft.fft2(padded_image))
    P0             = abs2(padded_image_f)
    PS_map0        = radial_average_map(P0,nbins)

#   First iteration of the normalised filter:

    filter0        = normalized_mf(gauss_beam_f,PS_map0)

#   First filtering:

    filtered_padded_image1_f = np.multiply(padded_image_f,filter0)
    filtered_padded_image1   = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_padded_image1_f)))
    filtered_image1          = filtered_padded_image1[l:l+size0,l:l+size0]

#   If iterative is False, then our filtered image is fiiltered_image1

    if not iterative:

        filtered_image = filtered_image1

    else:

#   Detection of the peaks above a cut in SNR,
#    fit to a Gaussian around that regions and substraction from the
#    input map

        peaks  = peak_local_max(filtered_image1,
                                min_distance=int(fwhm_pix),
                                threshold_abs=threshold*filtered_image1.std(),
                                exclude_border=int(fwhm_pix),
                                indices=True)
        npeaks = len(peaks)

        if npeaks<1:

            filtered_image   = filtered_image1  # It isn't necessary to iterate

            print(' ')
            print(' ---- No peaks above threshold in the filtered image ')
            print(' ')
            filter1=filter0


        else:

            print(' ')
            print(' ---- {0} peaks above threshold '.format(npeaks))
            print(' ')

            image1 = input_image.copy()

            for peak in range(npeaks):

#   We select a patch around that peak

                locat = peaks[peak,:]
                patch,xinf,xsup,yinf,ysup = poststamp(image1,
                                                      5.0*fwhm_pix,
                                                      locat)

#   We fit to a Gaussian profile with fixed width in that patch

                fitted_peak = fit_single_peak(patch,
                                              fixwidth=True,
                                              fixed_sigma=sigma_pix)

#   We subtract the fitted Gaussian from a copy of the original data

                image1[xinf:xsup,yinf:ysup] = image1[xinf:xsup,yinf:ysup] - fitted_peak.gaussmap


#   Second interation of the filter:

            l               = 2*int(fwhm_pix)
            padscheme       = ((l,l),(l,l))
            padded_image1   = np.pad(image1,
                                     padscheme,
                                     mode='reflect')

#   FFT and power spectrum map of the original data:

            padded_image1_f  = np.fft.fftshift(np.fft.fft2(padded_image1))
            P1               = abs2(padded_image1_f)
            PS_map1          = radial_average_map(P1,nbins)

#   Second iteration of the normalised filter:

            filter1          = normalized_mf(gauss_beam_f,
                                             PS_map1)

#   Second filtering:

            filtered_padded_image1_f = np.multiply(padded_image1_f,filter1)
            filtered_padded_image1   = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_padded_image1_f)))
            filtered_image           = filtered_padded_image1[l:l+size0,l:l+size0]

    return filtered_image,filter0


### --- FAST MATRIX ARITHMETICS -----------------------------------------------

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


### --- FILTER DESIGN ---------------------------------------------------------

def normalized_mf(fourier_profile,power_spectrum_map):
    """
    This assumes the source profile has amplitude == 1 in real space
    """

    pc                   = fourier_profile.shape[0]//2
    unnorm_filter        = np.divide(fourier_profile,
                                     power_spectrum_map)
    unnorm_filter[pc,pc] = 0.0

    fnorm                = np.multiply(fourier_profile,unnorm_filter)
    rnorm                = np.real(np.fft.ifft2(np.fft.ifftshift(fnorm)))
    normaliz             = np.amax(rnorm)
    norm_filter          = unnorm_filter/normaliz

    return norm_filter

### --- GAUSSIAN PROFILE ------------------------------------------------------

def Gaussian_profile(size,fwhm_pix=3):

    """
    Makes a square image of a Gaussian kernel, returned as a numpy
    two-dimensional array. The Gaussian takes a maximum value = 1
    located at the center of the image

    """

    sigma_pix = fwhm2sigma*fwhm_pix

    x0 = y0 = size // 2
    y  = np.arange(0,size,1,float)
    x  = y[:,np.newaxis]
    r2 = (x-x0)**2+(y-y0)**2

    g  = np.exp(-r2/(2*sigma_pix**2))

    return g


### --- RADIAL AVERAGE MAP  ---------------------------------------------------


def radial_average_map(array,nbins=50):

    s    = array.shape
    size = s[0]

    rvec = [None] * nbins
    avec = [None] * nbins

    x    = np.arange(0, size, 1, float)
    y    = x[:,np.newaxis]
    x0   = y0 = size // 2
    r    = np.sqrt((x-x0)**2+(y-y0)**2)

    rbin = (nbins * r/r.max()).astype(np.int)

    avec = ndimage.mean(array, labels=rbin, index=np.arange(0, rbin.max()))

    rvec = [i for i in range(nbins)]
    rvec = np.multiply(rvec,r.max())
    rvec = np.divide(rvec,float(nbins))+r.max()/(2.0*nbins)

    avec = [array[x0,y0]] + np.array(avec).tolist()
    rvec = [0] + np.array(rvec).tolist()

    f    = interp1d(rvec,avec,bounds_error=False,fill_value=avec[nbins-1])
    pmap = f(r)

    return pmap

### --- CUT POSTSTAMPS -------------------------------------------------------

def poststamp(parent_image,post_size,location):

    nx = ny = int(post_size)//2
    sz = parent_image.shape[0]

    xinf = int(location[0])-nx
    if xinf<0:
        xinf = 0
    xsup = int(location[0])+nx
    if xsup>(sz-1):
        xsup = sz-1

    yinf = int(location[1])-ny
    if yinf<0:
        yinf = 0
    ysup = int(location[1])+ny
    if ysup>(sz-1):
        ysup = sz-1

    stamp = parent_image[xinf:xsup,yinf:ysup]

    return stamp,xinf,xsup,yinf,ysup


### --- FIT TO TWO-DIEMENSIONAL GAUSSIANS  -------------------------------------

class Gaussfit:
    def __init__(self,model,amplitude,x,y,sigma,synthetic,gaussmap,residual):
        self.model = model
        self.amplitude = amplitude
        self.x = x
        self.y = y
        self.sigma = sigma
        self.synthetic = synthetic
        self.gaussmap = gaussmap
        self.residual = residual

def tiedfunc(g1):

    """
    Un ejemplo de parámetros 'tied'. En este caso, vamos a forzar a que el
    modelo gaussiano sea circularmente simétrico
    """
    y_stddev_1 = g1.x_stddev_1
    return y_stddev_1

def fit_single_peak(patch,toplot=False,fixwidth=False,fixed_sigma=2.0,
                    fixcenter=False,center=None):

    """
       Fits an image patch to a composite model consisting of
    a planar baseline plus a symmetric Gaussian. Returns a fit object
    of the Gaussfit class

    """

    m0 = patch.mean()
    a0 = patch.max()
    s0 = float(patch.shape[0])/8.0
    mm = np.where(patch==a0)
    a0 = a0 - m0
    if np.size(mm)==2:
        y0 = 0.5+float(mm[0])
        x0 = 0.5+float(mm[1])
    else:
        y0 = 0.5+float(mm[0][0])
        x0 = 0.5+float(mm[0][1])
    if fixcenter:
        if center is not None:
            x0 = center[0]
            y0 = center[1]
            a0 = patch.mean()
        else:
            x0 = patch.shape[0]//2 - 0.5
            y0 = patch.shape[1]//2 - 0.5
            a0 = patch.mean()

    y, x = np.mgrid[:patch.shape[0], :patch.shape[1]]


    model1 = models.Polynomial2D(degree=1)

    if fixwidth:
        if fixcenter:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=fixed_sigma,
                                       y_stddev=fixed_sigma,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_stddev':True,
                                              'x_mean':True,
                                              'y_mean':True})
        else:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=fixed_sigma,
                                       y_stddev=fixed_sigma,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_stddev':True})
    else:
        if fixcenter:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=s0,
                                       y_stddev=s0,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_mean':True,
                                              'y_mean':True})
        else:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=s0,
                                       y_stddev=s0,
                                       theta=0.0,
                                       fixed={'theta':True})

    modelo = model1+model2
    modelo.y_stddev_1.tied = tiedfunc

    fit_p  = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(modelo, x, y, patch)

    fit_amplitude = p.amplitude_1.value
    fit_x         = p.x_mean_1.value
    fit_y         = p.y_mean_1.value
    fit_sigma     = p.x_stddev_1.value

    sintetica     = p(x,y)
    residuo       = patch-sintetica
    modg          = models.Gaussian2D(amplitude=fit_amplitude,
                                      x_mean=fit_x,y_mean=fit_y,
                                      x_stddev=fit_sigma,y_stddev=fit_sigma,
                                      theta=0.0)
    gmap          = modg(x,y)

    f = Gaussfit(p,fit_amplitude,fit_x,fit_y,fit_sigma,sintetica,gmap,residuo)

    if toplot:

        plt.figure(figsize=(12,12))
        plt.subplot(221)
        plt.pcolormesh(patch)
        plt.axis('tight')
        plt.title('Data')
        plt.colorbar()

        plt.subplot(222)
        plt.pcolormesh(sintetica)
        plt.axis('tight')
        plt.title('Planar baseline + Gaussian model')
        plt.colorbar()

        modg = models.Gaussian2D(amplitude=fit_amplitude,
                                 x_mean=fit_x,y_mean=fit_y,x_stddev=fit_sigma,
                                 y_stddev=fit_sigma,theta=0.0)
        plt.subplot(223)
        plt.pcolormesh(gmap)
        plt.axis('tight')
        plt.title('Gaussian model')
        plt.colorbar()

        plt.subplot(224)
        plt.pcolormesh(residuo)
        plt.axis('tight')
        plt.title('Residual')
        plt.colorbar()

    return f