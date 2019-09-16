# -*- coding: utf-8 -*-

"""
Created on Wed Jul  3 14:19:34 2019

@author: david
"""


import healpy as hp
import numpy as np
import sep
import os
from pathlib import Path
parent_path=os.fspath(Path(__file__).parent.parent)
data_path=parent_path+'/5mindata/npymats'

def generate_sim_2048(fwhmminutes, ps_number, output_file):
    """
    Generates a map of the CMB, noise and spreads a number of random point 
    sources. Outputs a .fits map and a .txt with the location of the sources
    and their amplitude. Two-file header.
    
    Input:
        fwhmminutes, fwhm of the instrument in arcminutes
        mean_ps_number, mean number of point sources in the map. The number is
            random, following a normal distribution with variance
        var_ps_number, variance of this normal distribution
        ps_strength, mean amplitude of the ps sources. The amplitudes follow
            a normal distribution with variance
        ps_variance, variance of this normal distribution
        output_file, name of the output .fits and .txt files
    """
    
    #Approximately 90000 PS in the sky with these parameters
    
    noise_factor=19.4 #gaussian noise, per pixel
    noisemap=np.random.normal(loc=0.0,scale=noise_factor,size=(12*2048*2048))

    data=np.loadtxt('COM_PWR_CMB.txt', skiprows=1)
    
    fwhm_143=fwhmminutes*np.pi/(180*60)
    nside=2048
    #sigma=(1.3*10**-6/fwhm)

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
        
    map_cmb=hp.sphtfunc.synfast(cls, nside, pixwin=True, fwhm=fwhm_143)
    
    #POINT-LIKE SOURCE CREATION
    
    xo=-0.9789
    alpha=-2.193
    beta=np.e**(alpha*(-xo-4.5))
    k=np.e**(alpha*(-xo-2))-beta
    r=10**xo
    
    positions=np.random.random_integers(0,high=len(map_cmb), size=ps_number)
    y=np.random.random_sample(size=ps_number)
    map_ps=np.zeros((nside*nside*12,))
    amplitudes=r*10**(6+np.log(k*y+beta)/alpha)#el 6 corresponde a uK-K
    
    for i in range(ps_number):
        map_ps[positions[i]]=map_ps[positions[i]]+amplitudes[i]
        
    pssmth=hp.smoothing(map_ps, fwhm=fwhm_143)
    hp.mollview(pssmth)
    
    smthfinal=pssmth+map_cmb
    hp.mollview(smthfinal+noisemap)
    
    alms=hp.map2alm(smthfinal+noisemap)
    cl_noise=noise_factor*noise_factor*4*np.pi/(12*nside*nside)
    wl=np.exp(-l*(l+1)*0.5*(fwhm_143/2.355)**2)
    cl_theo=cls*wl*wl+cl_noise
    matched_filter=wl/cl_theo
    
    alms_mf=hp.almxfl(alms, matched_filter)

    map_filtered=hp.alm2map(alms_mf,nside) 
      
    if output_file!='':
        hp.fitsfunc.write_map(output_file+'.fits', smthfinal+noisemap)
        hp.fitsfunc.write_map(output_file+'_pssmth.fits', pssmth)
        hp.fitsfunc.write_map(output_file+'_mfiltered.fits', map_filtered)


def generate_training_matrices(mapp, psmap, mfmap, pix_index, nside, picind, path):
    """
    """
    size=150
    res=1.7
    path=path+'/npymats'
    
    #hp.reorder(mapp, r2n=True)
    lon,lat=hp.pix2ang(nside, pix_index, nest=False, lonlat=True)
    fullimg=np.array(hp.visufunc.gnomview(map=mapp, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
    psimg=np.array(hp.visufunc.gnomview(map=psmap, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
    mfimg=np.array(hp.visufunc.gnomview(map=mfmap, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
    
    np.save(path+'/full_'+str(picind)+'_'+str(pix_index)+'.npy', fullimg)
    np.save(path+'/smthps_'+str(picind)+'_'+str(pix_index)+'.npy', psimg)
    np.save(path+'/mfiltered_'+str(picind)+'_'+str(pix_index)+'.npy', mfimg)
    
    objects=sep.extract(psimg, 2, filter_kernel=None)
    
    segment=np.zeros((size+4,size+4))
    for ps in objects:
        i=objects['ycpeak']+2
        j=objects['xcpeak']+2
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

    np.save(path+'/segps_'+str(picind)+'_'+str(pix_index)+'.npy', segment[2:size+2,2:size+2])
    

def generate_training_set():
    
    path=parent_path+'/5mindata'
    
#    noise1=hp.fitsfunc.read_map('./ffp10_noise_143_full_map_mc_00001.fits')*1e6
#    noise2=hp.fitsfunc.read_map('./ffp10_noise_143_full_map_mc_00002.fits')*1e6
#    
#    generate_sim_2048(7.22, 100000, noise1, './fullmaps_7/sim_1')
#    generate_sim_2048(7.22, 80000, noise2, './fullmaps_7/sim_2')
#    generate_sim_2048(7.22, 85000, noise2, './fullmaps_7/sim_3')
#    noise2=None
    
#    

#    generate_sim_2048(5, 85000, path+'/fullmaps/sim_1')
#    generate_sim_2048(5, 105000, path+'/fullmaps/sim_2')
#    generate_sim_2048(5, 95000, path+'/fullmaps/sim_3')
    
    for i in [3]:
        mapp=hp.fitsfunc.read_map(path+'/fullmaps/'+'sim_'+str(i)+'.fits', nest=False)
        psmap=hp.fitsfunc.read_map(path+'/fullmaps/'+'sim_'+str(i)+'_pssmth.fits', nest=False)
        mfmap=hp.fitsfunc.read_map(path+'/fullmaps/'+'sim_'+str(i)+'_mfiltered.fits', nest=False)
        for j in range(1900,3072):
            generate_training_matrices(mapp, psmap, mfmap, j, 16, i, path)
            if j%100==0:
                print(str((3-i)*3072-j-1))
        mapp=None
        psmap=None

generate_training_set()