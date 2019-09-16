# -*- coding: utf-8 -*-

"""
Created on Thu Jul  4 12:59:23 2019

@author: david
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


def image_loading(lenweak,foreorder):
    size=150
    val_full_strong=[]
    val_full_weak=[]
    positions_weak=[]
    positions_strong=[]
    
    for i in range(lenweak):
        rem=foreorder[i]
        posmap=np.load(parent_path+'/5minplanedata/npymats/location_'+str(rem)+'.npy').astype(np.int32)
        positions_weak.append(np.array(posmap))
        full=np.load(parent_path+'/7mindata/real_planck/full_'+str(rem)+'.npy')*1e6
        val_full_weak.append(np.array(full))
    for i in range(lenweak,3072):
        rem=foreorder[i]
        posmap=np.load(parent_path+'/5minplanedata/npymats/location_'+str(rem)+'.npy').astype(np.int32)
        positions_strong.append(np.array(posmap))
        full=np.load(parent_path+'/7mindata/real_planck/full_'+str(rem)+'.npy')*1e6
        val_full_strong.append(np.array(full))
        
    modelweak=load_model('model75_weakforegrounds_14ep')
    modelstrong=load_model('model75_strongforegrounds_24ep')
    
    val_full_weak=np.array(val_full_weak).reshape(lenweak,size,size,1)
    val_full_strong=np.array(val_full_strong).reshape(3072-lenweak,size,size,1) 
    val_predictions_weak=modelweak.predict(val_full_weak)
    val_predictions_strong=modelstrong.predict(val_full_strong)
    
    
    print('Images loaded')
    return(val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, positions_weak, positions_strong)
    
def draw_globalmap_cnn(threshold, set_predictions, set_locations):
    positionalmap=np.zeros((2048*2048*12),dtype=np.int8) #1 if detection
    for i in range(len(set_predictions)):
        pos=img_acc_cnn_locations(threshold, set_predictions[i,:,:,0], set_locations[i])
        #pos_xxx es una lista de arrays 7x7 con las posiciones correspondientes
        # Pinta 0s donde no hay fuentes o hay espúreas, y las pone nuevas
        for mat in pos:
            for x in range(7):
                for y in range(7):
                    positionalmap[mat[x,y]]=0
            positionalmap[mat[3,3]]=1

    return positionalmap
    
def img_acc_cnn_locations(threshold, img, locationmap):
    """
    Returns:
        number of objects in the label
        number of true detection
        number of false detection
    Input:
        img - image to process in greyscale before thresholding has been applied
        labelimage - image with the labels, 0 or 255 scale.
    """
    objectsimg,segmap_img=sep.extract(img, threshold, segmentation_map=True, filter_kernel=None)
    xcpeak=objectsimg['xcpeak']
    ycpeak=objectsimg['ycpeak']
    
    pos=[]
                    
    for i in range(len(objectsimg)):
        xsep=xcpeak[i]
        ysep=ycpeak[i]
        if ysep>4 and ysep<145 and xsep>4 and xsep<145:
            pos.append(locationmap[ysep-3:ysep+4,xsep-3:xsep+4])    
    return pos

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
    ps_x=[]
    ps_y=[]
    
    for j in range(np.shape(positionalmap)[0]):
        if positionalmap[j]==1:
            ps_x.append(hp.pix2ang(2048,j)[0])
            ps_y.append(hp.pix2ang(2048,j)[1])
    
    hp.mollview(foregrounds,cmap=plt.get_cmap('Blues'),max=4,min=-0.2,title='')#, title='PS detection in map, thrs='+str(threshold))
    hp.projscatter(ps_x,ps_y,s=5,color='green')
    hp.visufunc.graticule(dpar=15, dmer=30)
    print('Number of detections: '+str(len(ps_x)))
    
    plt.savefig('mollplanck75-98.png', format='png', dpi=300)


threshold=0.23
foreorder=np.load('fore_order.npy')
lenweak=int(0.75*3072)

#val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_locations_weak, val_locations_strong=image_loading(lenweak,foreorder)

#val_predictions=np.concatenate((val_full_weak,val_full_strong))

positionalmap=draw_globalmap_cnn(threshold, val_predictions_weak, val_locations_weak)
positionalmap=positionalmap.astype(float)
#hp.mollview(positionalmap)
#foremap=np.zeros((12*12*2048))
#foremap=hp.read_map(parent_path+'/5minplanedata/fullmaps/foreground_psm.fits')
#numberedmap=create_foremap_numbered(foremap*1e6)

plot_map_withsources(positionalmap,numberedmap)