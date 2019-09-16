# -*- coding: utf-8 -*-

"""
This script and method library evaluates the performance of the CNN and the 
MF and compares them, for point sources in the plane.

Version: August 18, 2019

@author: David Balbas
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sep
import os
from pathlib import Path
parent_path=os.fspath(Path(__file__).parent.parent)
data_path=parent_path+'/7mindata/npymats'

#Lists to characterize the point sources that are detected. Comment if not needed.
mfilter_detections_spur_list=[] #elements: [detected flux, original flux,spur=1,true=0]
mfilter_detections_true_list=[]
cnn_detections_spur_list=[] #elements: [detected flux, original flux,spur=1,true=0]
cnn_detections_true_list=[]

def image_loading(num):  
    size=150
    nmaps=1
    val_full=[]
    val_label=[]
    val_pssmth=[]
        
    lenj=3072*nmaps-num #Number of images to be evaluated.
    indices=np.arange(3072*nmaps)
    np.random.shuffle(indices)
    valindices=indices[lenj:]
            
    for j in valindices:
        i=1+j//3072 #Change index for changing the loaded map
        rem=j%3072
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        val_full.append(np.array(full))
        pssmth=np.load(data_path+'/smthps_'+str(i)+'_'+str(rem)+'.npy')
        val_pssmth.append(np.array(pssmth))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        val_label.append(np.array(label))
    
    #Up to this point: arrays loaded in val_(full,label,mfiltered).
        
    model=load_model('cnn7min12ep')
    model2=load_model('cnn2_7min12ep')
    
    val_full=np.array(val_full).reshape(3072*nmaps-lenj,size,size,1)
    print('start')
    val_predictions=model.predict(val_full)
    val_predictions2=model2.predict(val_full)
    print('end')

    return (val_predictions, val_predictions2 ,val_pssmth,val_full,val_label)

def img_accuracy_cnn_2(threshold, img, labelimg, full_img, flux, labelmap):
    """
    Evaluates the accuracy of the CNN in a single image
    """
    objectsimg,segmap_img=sep.extract(img, threshold, segmentation_map=True, filter_kernel=None)
    objectslabel,segmap_label=sep.extract(labelimg, 1, segmentation_map=True, filter_kernel=None)
    xcpeak=objectsimg['xcpeak']
    ycpeak=objectsimg['ycpeak']
    peak=objectslabel['peak']
    label_xcpeak=objectslabel['xpeak']
    label_ycpeak=objectslabel['ypeak']
    image_peak=objectsimg['peak']
    
    above_detections=0
    below_detections=0
    undetected_above=0
    undetected_below=0
    real_objects=0
    spur_objects=0
    
    for i in range(len(objectslabel)):
        xsep=label_xcpeak[i]
        ysep=label_ycpeak[i]
        pk=peak[i]
        if ysep>4 and ysep<145 and xsep>4 and xsep<145:
            if np.amax([segmap_img[ysep-2:ysep+3,xsep-2:xsep+3]])>0.1:
                if pk>flux:
                    above_detections=above_detections+1
                else:
                    below_detections=below_detections+1
            else:
                if pk>flux:
                    undetected_above=undetected_above+1
                else:
                    undetected_below=undetected_below+1
                    
    for i in range(len(objectsimg)):
        xsep=xcpeak[i]
        ysep=ycpeak[i]
        ipk=image_peak[i]
        if ysep>4 and ysep<145 and xsep>4 and xsep<145:
            if np.amax([labelmap[ysep-2:ysep+3,xsep-2:xsep+3]])>0.1:
                real_objects=real_objects+1
                maxarray=np.amax(labelimg[ysep-2:ysep+3,xsep-2:xsep+3])
                maxarray2=np.amax([full_img[ysep-2:ysep+3,xsep-2:xsep+3]])
                cnn_detections_true_list.append([ipk, maxarray2, maxarray])
            else:
                spur_objects=spur_objects+1
                maxarray=np.amax([full_img[ysep-2:ysep+3,xsep-2:xsep+3]])
                cnn_detections_spur_list.append([ipk, maxarray])
    
    return (above_detections,below_detections,undetected_above,undetected_below,real_objects,spur_objects)


def total_comparator(val_mfiltered, val_predictions, val_ps, val_full, val_label):
    """
    Compares the performance of the CNN and the MF. Change the internal parameters
    of the method if desired.
    """
    
    #above_detections,below_detections,undetected_above,undetected_below,real_objects,spur_objects)
    val_mfiltered=np.array(val_mfiltered).reshape(len(val_ps),150,150,1)
    threshold=[0.105,0.135,0.26] #CNN thresholds to be evaluated.
    threshold2=[0.08,0.11,0.21] #MF thresholds to be evaluated (in sigmas)
    num=40 #number of fluxes to be evaluated
    fluxes=np.linspace(1,400,num=num) #Flux in Tps
    criticalflux=410 #if a non-detection occurs above this Tps, prints the image.
    cnn_completeness=np.zeros((len(threshold),num))
    cnn_spures=np.zeros((len(threshold)))
    mfil_completeness=np.zeros((len(threshold2),num))
    mfil_spures=np.zeros((len(threshold2)))
    for thr_ind in range(len(threshold)):
        thr=threshold[thr_ind]
        above_detections=np.zeros((num))
        below_detections=np.zeros((num))
        undetected_above=np.zeros((num))
        undetected_below=np.zeros((num))
        real_objects=np.zeros((num))
        spur_objects=np.zeros((num))
        for ind in range(num):
            flux=fluxes[ind]
            for i in range(len(val_predictions)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions[i,:,:,0],val_ps[i],val_full[i,:,:,0],flux,val_label[i])
                
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                
                if flux>criticalflux and ua>0:
                    plt.figure()
                    plt.pcolormesh(val_predictions[i,:,:,0])
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.pcolormesh(val_full[i,:,:,0])
                    plt.show()
                    plt.close()
                    print(i)
                    
        cnn_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        cnn_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    
    fluxes2=fluxes
        
    for thr_ind in range(len(threshold2)):
        thr=threshold2[thr_ind]
        above_detections=np.zeros((num))
        below_detections=np.zeros((num))
        undetected_above=np.zeros((num))
        undetected_below=np.zeros((num))
        real_objects=np.zeros((num))
        spur_objects=np.zeros((num))
        for ind in range(num):
            flux=fluxes2[ind]
            for i in range(len(val_predictions)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_mfiltered[i,:,:,0],val_ps[i], val_full[i,:,:,0],flux,val_label[i])
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                
                if flux>criticalflux and ua>0:
                    plt.figure()
                    plt.pcolormesh(val_mfiltered[i,:,:,0])
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.pcolormesh(val_full[i,:,:,0])
                    plt.show()
                    plt.close()
                    print(i)
                    
        mfil_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        mfil_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    
    #Comparison finished. Plot the basic results.
    plt.figure()
    colours=['red','blue','green','black']
    for thr_ind in range(len(threshold)):
        thr=threshold[thr_ind]
        plt.plot(fluxes*0.001920, cnn_completeness[thr_ind,:], label='old CNN '+str(thr), color=colours[thr_ind], linestyle='dashed')
        print("Spureous ratio, old CNN at threshold "+str(thr)+": "+str(cnn_spures[thr_ind]))
    for thr_ind in range(len(threshold2)):
        thr=threshold2[thr_ind]
        plt.plot(fluxes*0.001920, mfil_completeness[thr_ind,:], label='new CNN '+str(thr), color=colours[thr_ind])
        print("Spureous ratio, new CNN at threshold "+str(thr)+": "+str(mfil_spures[thr_ind]))
    plt.xlabel('S / Jy')
    plt.ylabel('Completeness')
    plt.legend()
    plt.savefig('fullcompl7min2comp.svg',format='svg')
    plt.savefig('fullcompl7min2comp.png',format='png', dpi=300)
    plt.show()
    plt.close()
    return (above_detections,below_detections,real_objects,spur_objects,fluxes, cnn_completeness, mfil_completeness, cnn_spures, mfil_spures)

#obtains a full comparison of the matched filter and the CNN
    
#val_predictions,val_predictions2,val_pssmth,val_full,val_label=image_loading(3072)

results2=total_comparator(val_predictions2, val_predictions,val_pssmth,val_full,val_label)

#results3=total_comparator(val_mfiltered,val_predictions,val_pssmth,val_full,val_label)

##SCRIPT FOR STUDYING SOURCES AND DETECTIONS. COMMENT IF NOT NEEDED.

cnn_detections_spur_list=np.array(cnn_detections_spur_list)
cnn_detections_true_list=np.array(cnn_detections_true_list)
#
##ORDER: Significancy, Ttotalmap, Tps
#
#plt.figure(1)
##plt.title('Flux estimation MF')
#plt.scatter(mfilter_detections_true_list[:,2]*0.0009212, mfilter_detections_true_list[:,0], s=3, color='green')
##plt.scatter(mfilter_detections_spur_list[:,1], mfilter_detections_spur_list[:,0], s=4, color='red')
#plt.xlabel('S / Jy')
#plt.ylabel(r'MF prediction / $\sigma$')
#plt.savefig('mfdetectiondistr.png',format='png', dpi=300)
#plt.show()
#plt.close()
#
#
##plt.figure(2)
##plt.title('True MF over T ps')
##plt.scatter(mfilter_detections_true_list[:,2], mfilter_detections_true_list[:,0], s=4, color='green')
##plt.xlabel('T ps')
##plt.ylabel(r'Significancia ($\sigma$)')
##plt.show()
##plt.close()
#
#plt.figure(3)
##plt.title('Distribution of real detections CNN')
#plt.scatter(cnn_detections_true_list[:,1], cnn_detections_true_list[:,0], s=3, color='green')
#plt.scatter(cnn_detections_spur_list[:,1], cnn_detections_spur_list[:,0], s=4, color='red')
#plt.xlim(-200,5200)
#plt.xlabel(r'$T_{map}$ / $\mu$K')
#plt.ylabel(r'CNN prediction')
#plt.savefig('cnndetectiondistr.png',format='png', dpi=300)
#plt.show()
#plt.close()
