#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module divides the sky in two different areas for the detection with two separate CNN models. Completenes curves are not shown, only detection maps.

@version: 17 September 2019
@author: David Balbas
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sep
import healpy as hp
import os
from pathlib import Path
parent_path=os.fspath(Path(__file__).parent.parent)
data_path=parent_path+'/7mindata/npymats'

def image_loading(num,lenweak):  
    size=150
    foreorder=np.load('fore_order.npy') #weak foregrounds go FIRST
    
    val_full_strong=[]
    val_full_weak=[]
    val_pssmth_strong=[]
    val_pssmth_weak=[]
    positions_weak=[]
    positions_strong=[]
    val_label_weak=[]
    val_label_strong=[]
    
    i=num
    for j in range(lenweak):
        rem=foreorder[j]
        posmap=np.load(parent_path+'/5minplanedata/npymats/location_'+str(rem)+'.npy').astype(np.int32)
        positions_weak.append(np.array(posmap))
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        val_full_weak.append(np.array(full))
        pssmth=np.load(data_path+'/smthps_'+str(i)+'_'+str(rem)+'.npy')
        val_pssmth_weak.append(np.array(pssmth))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        val_label_weak.append(np.array(label))
    for j in range(lenweak,3072):
        rem=foreorder[j]
        posmap=np.load(parent_path+'/5minplanedata/npymats/location_'+str(rem)+'.npy').astype(np.int32)
        positions_strong.append(np.array(posmap))
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        val_full_strong.append(np.array(full))
        pssmth=np.load(data_path+'/smthps_'+str(i)+'_'+str(rem)+'.npy')
        val_pssmth_strong.append(np.array(pssmth))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        val_label_strong.append(np.array(label))
        
    modelweak=load_model('model75_weakforegrounds_14ep')
    modelstrong=load_model('model75_strongforegrounds_24ep')
    
    val_full_weak=np.array(val_full_weak).reshape(lenweak,size,size,1)
    val_full_strong=np.array(val_full_strong).reshape(3072-lenweak,size,size,1) 
    val_predictions_weak=modelweak.predict(val_full_weak)
    val_predictions_strong=modelstrong.predict(val_full_strong)
    
    print('Images loaded')
    return(val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_label_weak, val_label_strong, val_pssmth_weak,val_pssmth_strong, positions_weak, positions_strong)
    
def draw_globalmap_cnn(threshold, flux, set_full, set_predictions, set_pssmth, set_label, set_locations):
    """
    Creates the map with the CNN detections, spurious and non-detected sources
    """
    positionalmap=np.zeros((2048*2048*12),dtype=np.int8) #-1 if spur>threshold, 1 if undetected>flux, 2 if detected>flux
    above_detections=0
    below_detections=0
    undetected_below=0
    undetected_above=0
    real_objects=0
    spur_objects=0
    for i in range(len(set_predictions)):
        ad,bd,ua,ub,ro,so, pos_spurs, pos_correct, pos_undetected=img_acc_cnn_locations(
                threshold, flux, set_predictions[i,:,:,0], set_pssmth[i], set_full[i,:,:,0], set_label[i], set_locations[i])
        above_detections=above_detections+ad
        below_detections=below_detections+bd
        undetected_below=undetected_below+ub
        undetected_above=undetected_above+ua
        real_objects=real_objects+ro
        spur_objects=spur_objects+so
        #pos_xxx is a list of 7x7 arrays with the corresponding positions.
        #it writes 0s where there are no sources or there are spureous detections,
        # and it places new arrays.
        for mat in pos_spurs:
            for x in range(7):
                for y in range(7):
                    if positionalmap[mat[x,y]] <0:
                        positionalmap[mat[x,y]]=0
            positionalmap[mat[3,3]]=-1
        # Paints 0s surrounding undetected sources
        for mat in pos_undetected:
            flag=False
            for x in range(7):
                for y in range(7):
                    if positionalmap[mat[x,y]]==1:
                        positionalmap[mat[x,y]]=0
                    if positionalmap[mat[x,y]]==2:
                        flag=True
            if flag==False:          
                positionalmap[mat[3,3]]=1
        for mat in pos_correct:
            for x in range(7):
                for y in range(7):
                    positionalmap[mat[x,y]]=0
            positionalmap[mat[3,3]]=2

    return (positionalmap, above_detections, below_detections, undetected_below, undetected_above, real_objects, spur_objects)

    
def img_acc_cnn_locations(threshold, flux, img, labelimg, full_img, labelmap, locationmap):
    """
    Evaluates the accuracy of the CNN and returns the positions of the detections.
    """
    objectsimg,segmap_img=sep.extract(img, threshold, segmentation_map=True, filter_kernel=None)
    objectslabel,segmap_label=sep.extract(labelimg, 1, segmentation_map=True, filter_kernel=None)
    xcpeak=objectsimg['xcpeak']
    ycpeak=objectsimg['ycpeak']
    peak=objectslabel['peak']
    label_xcpeak=objectslabel['xpeak']
    label_ycpeak=objectslabel['ypeak']
    image_peak=objectsimg['peak']
    
    pos_spurs=[]
    pos_undetected=[]
    pos_correct=[]
    ad=0
    bd=0
    ua=0
    ub=0
    ro=0
    so=0
    
    for i in range(len(objectslabel)):
        xsep=label_xcpeak[i]
        ysep=label_ycpeak[i]
        pk=peak[i]
        if ysep>4 and ysep<145 and xsep>4 and xsep<145:
            if np.amax([segmap_img[ysep-2:ysep+3,xsep-2:xsep+3]])>0.1:
                pos_correct.append(locationmap[ysep-3:ysep+4,xsep-3:xsep+4])
                if pk>flux:
                    ad=ad+1
                    #pos_correct....
                else:
                    bd=bd+1
            else:
                if pk>flux:
                    ua=ua+1
                    pos_undetected.append(locationmap[ysep-3:ysep+4,xsep-3:xsep+4])
                else:
                    ub=ub+1
                    
    for i in range(len(objectsimg)):
        xsep=xcpeak[i]
        ysep=ycpeak[i]
        ipk=image_peak[i]
        if ysep>4 and ysep<145 and xsep>4 and xsep<145:
            if np.amax([labelmap[ysep-2:ysep+3,xsep-2:xsep+3]])>0.1:
                ro=ro+1
                #maxarray=np.amax(labelimg[ysep-2:ysep+3,xsep-2:xsep+3])
                #maxarray2=np.amax([full_img[ysep-2:ysep+3,xsep-2:xsep+3]])
                #cnn_detections_true_list.append([ipk, maxarray2, maxarray])
            else:
                so=so+1
                pos_spurs.append(locationmap[ysep-3:ysep+4,xsep-3:xsep+4])
                #maxarray=np.amax([full_img[ysep-2:ysep+3,xsep-2:xsep+3]])
                #cnn_detections_spur_list.append([ipk, maxarray])
    
    return (ad,bd,ua,ub,ro,so,pos_spurs,pos_correct,pos_undetected)

def create_foremap_numbered(foremap):
    """
    Creates a region-divided foreground map according to the foreground
    intensity at each region. Not required for the evaluation, only for
    visualization purposes
    """
    percentilemap=np.zeros(np.size(foremap))
    for i in range(len(foremap)):
        if foremap[i]>2000:
            percentilemap[i]=4
        elif foremap[i]>500:
            percentilemap[i]=3
        elif foremap[i]>200:
            percentilemap[i]=2
        elif foremap[i]>70:
            percentilemap[i]=1
    return percentilemap

def plot_map_withsources(positionalmap,foregrounds):
    """
    Creates and plots the map where point sources (detections, undetected
    and spurious) are plotted in their respective positions.
    """
    true_ps_x=[]
    spur_ps_x=[]
    nondetected_ps_x=[]
    true_ps_y=[]
    spur_ps_y=[]
    nondetected_ps_y=[]
    for j in range(np.shape(positionalmap)[0]):
        if positionalmap[j]==2:
            true_ps_x.append(hp.pix2ang(2048,j)[0])
            true_ps_y.append(hp.pix2ang(2048,j)[1])
        if positionalmap[j]==1:
            nondetected_ps_x.append(hp.pix2ang(2048,j)[0])
            nondetected_ps_y.append(hp.pix2ang(2048,j)[1])
        if positionalmap[j]==-1:
            spur_ps_x.append(hp.pix2ang(2048,j)[0])
            spur_ps_y.append(hp.pix2ang(2048,j)[1])
    
    hp.mollview(foregrounds,cmap=plt.get_cmap('Blues'),max=4,min=-0.2,title='')#, title='PS location, t='+str(threshold)+r',$ S_{min}=$'+str(fluxjy)+' Jy')
    hp.projscatter(true_ps_x,true_ps_y,s=5,color='green')
    hp.projscatter(spur_ps_x,spur_ps_y,s=5,color='red')
    hp.visufunc.graticule(dpar=15, dmer=30)
    hp.projscatter(nondetected_ps_x,nondetected_ps_y,s=5,color='yellow')
    plt.savefig('mollweide75weak.svg', format='svg')
    plt.savefig('mollweide75weak.png', format='png', dpi=300)
    
    print('Number of true detections: '+str(len(true_ps_x)))
    print('Number of spurious: '+str(len(spur_ps_x)))
    print('Number of non-detected: '+str(len(nondetected_ps_x)))
    print('Completeness: '+str(len(true_ps_x)/(len(true_ps_x)+len(nondetected_ps_x))))
    
def locationmap_generator():
    locarray=np.zeros((12*2048*2048))
    for i in range(12*2048*2048):
        locarray[i]=i
    res=1.7
    size=150
    for i in range(3072):
        lon,lat=hp.pix2ang(16, i, nest=False, lonlat=True)
        locimg=np.array(hp.visufunc.gnomview(map=locarray, fig=None, rot=(lon,lat,0), xsize=size, ysize=size, reso=res, title='Gnomonic view', nest=False, return_projected_map=True, no_plot=True))
        np.save(data_path+'/location_'+str(i)+'.npy', locimg)
        if i%100==0:
            print(str(i))
            
threshold=0.15
fluxjy=0.200
flux=fluxjy/0.001920
val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_label_weak, val_label_strong, val_pssmth_weak,val_pssmth_strong, positions_weak, positions_strong=image_loading(1, int(0.75*3072))
positionalmap, above_detections, below_detections, undetected_below, undetected_above, real_objects, spur_objects=draw_globalmap_cnn(threshold, flux, val_full_weak, val_predictions_weak, val_pssmth_weak, val_label_weak, positions_weak)
##can include _fore
positionalmap=positionalmap.astype(float)
##hp.mollview(positionalmap)
##Create a foreground-numbered map if desired
foremap=hp.read_map(parent_path+'/5minplanedata/fullmaps/foreground_psm.fits')
numberedmap=create_foremap_numbered(foremap*1e6)
plot_map_withsources(positionalmap, numberedmap)
