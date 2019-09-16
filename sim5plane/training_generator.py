# -*- coding: utf-8 -*-

"""
Training generator, 5 arcminutes. PS thrown directly on plane
Version 18 August 2019

@author: david
"""
import healpy as hp
import numpy as np
from scipy.signal import convolve2d
import matched_filter_herranz as mfd
from scipy.ndimage import gaussian_filter as gaussker
from pathlib import Path
import os
parent_path=os.fspath(Path(__file__).parent.parent)
data_path=parent_path+'/5minplanedata'

def generate_simplecmb(fwhmminutes, output_file):
    """
    Generates a clean map of the CMB. Outputs a .fits map.
    """
    
    data=np.loadtxt('COM_PWR_CMB.txt', skiprows=1)
    
    fwhm=fwhmminutes*np.pi/(180*60)
    nside=2048
    
    #MAP AND NOISE SIMULATION
    l=np.zeros((7000,))#2509
    cls=np.zeros((7000,))
    cls[0]=0
    cls[1]=0
    for i in range(len(l)):
        l[i]=i
    
    cl=data[:,1]*2*np.pi/((data[:,0]+1)*(data[:,0])) #¿falta dividir por Tcmb²?
    
    for i in range(len(cl)):
        cls[i+2]=cl[i]
        
    map_cmb=hp.sphtfunc.synfast(cls, nside, pixwin=True, fwhm=fwhm)

    if output_file!='':
        hp.fitsfunc.write_map(output_file+'.fits', map_cmb)
    
def generate_training_matrices(mapp, foremapp, pix_index, nside, picind, path, ps_number, positions, amplitudes):
    """
    Generates a set of input training/validation arrays from a CMB clean simulation.
    """
    size=150
    res=1.7
    fwhm_pixel=5/res
    
    lon,lat=hp.pix2ang(nside, pix_index, nest=False, lonlat=True)
    fullimg=np.array(hp.visufunc.gnomview(map=mapp, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
    fore=np.array(hp.visufunc.gnomview(map=foremapp, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
    
    #PASO 1: Add point sources according to given positions and amplitudes
    ps=np.zeros((size*size))
    for ind in range(len(positions)):
        ps[positions[ind]]=amplitudes[ind]
    
    #PASO 2: Carry out a real-space smoothing.
    ps=ps.reshape(size,size)
    psimg=gaussker(ps, sigma=fwhm_pixel/2.355)
    finalimg=psimg+fullimg
    finalimg_fore=finalimg+fore
    
    #STEP 3: Adding white gaussian noise to the map
    noise_factor=19.4
    noise=np.random.normal(loc=0.0, scale=noise_factor, size=size*size).reshape(size,size)
    finalimg=finalimg+noise
    finalimg_fore=finalimg_fore+noise
    
    #STEP 4: Matched-filter the image in real space (and in fourier space,
    #       although this is not used later)
    mfarr,kernel=mfd.matched_filter(gaussker(finalimg,sigma=fwhm_pixel/2.355), fwhm_pix = fwhm_pixel, bins= 'auto')
    mfarr_fore,kernel_fore=mfd.matched_filter(gaussker(finalimg_fore,sigma=fwhm_pixel/2.355), fwhm_pix = fwhm_pixel, bins= 'auto')
    kernel=np.real(np.fft.fftshift(np.fft.ifft2(kernel)))[60:99,60:99]
    kernel_fore=np.real(np.fft.fftshift(np.fft.ifft2(kernel_fore)))[60:99,60:99]
        
    np.save(path+'full_'+str(picind)+'_'+str(pix_index)+'.npy', finalimg)
    np.save(path+'full_fore_'+str(picind)+'_'+str(pix_index)+'.npy', finalimg_fore)
    np.save(path+'mfiltered_'+str(picind)+'_'+str(pix_index)+'.npy', convolve2d(gaussker(finalimg,sigma=fwhm_pixel/2.355), kernel, mode='same',boundary='symm'))
    np.save(path+'mfiltered_fore_'+str(picind)+'_'+str(pix_index)+'.npy', convolve2d(gaussker(finalimg_fore,sigma=fwhm_pixel/2.355), kernel_fore, mode='same',boundary='symm'))
    np.save(path+'ps_'+str(picind)+'_'+str(pix_index)+'.npy', psimg)
    
    segmented=ps
    segment=np.zeros((size+4,size+4))
    for i in range(2,size+2):
        for j in range(2,size+2):
            if segmented[i-2,j-2]>0:
                #central cross
                segment[i,j]=1
                segment[i+1,j]=1
                segment[i,j+1]=1
                segment[i,j-1]=1
                segment[i-1,j]=1 
                #edges, square
                segment[i+1,j-1]=1
                segment[i-1,j+1]=1
                segment[i+1,j+1]=1
                segment[i-1,j-1]=1
                #further cross, 3x3
                segment[i+2,j]=1
                segment[i-2,j]=1
                segment[i,j+2]=1
                segment[i,j-2]=1

    np.save(path+'segps_'+str(picind)+'_'+str(pix_index)+'.npy', segment[2:size+2,2:size+2])

def generate_trainingset():
    """
    Generates a training set. The method shall be modified according to
    user requirements.
    """
    
    #generate_simplecmb(5, data_path+'/fullmaps/sim_1plane')
    
    foremap=hp.fitsfunc.read_map(data_path+'/fullmaps/foreground_psm.fits')*1e6
    
    xo=-0.9789
    alpha=-2.193
    beta=np.e**(alpha*(-xo-4.5))
    k=np.e**(alpha*(-xo-2))-beta
    r=10**xo
    
    path=data_path+'/npymats/'
    #CHANGE THE PATH
    i=0
    for ps_number in [100000]:
        i=i+1
        mapp=hp.fitsfunc.read_map(data_path+'/fullmaps/'+'sim_'+str(i)+'plane.fits', nest=False)
        npix=150*150*3072
        positions=np.random.random_integers(0,high=npix-1, size=ps_number)
        positions.sort()
        y=np.random.random_sample(size=ps_number)
        amplitudes=r*10**(6+np.log(k*y+beta)/alpha) #el 6 corresponde a uK-K
        fin=0
        for j in range(0,3072):
            ini=fin
            while fin<ps_number and positions[fin]<(j+1)*150*150:
                fin=fin+1
            generate_training_matrices(mapp, foremap, j, 16, i, path, ps_number, positions[ini:fin]-j*150*150, amplitudes[ini:fin])
            print(str((3-i)*3072-j-1))