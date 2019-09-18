#!/usr/bin/env python3
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
        i=3+j//3072 #Change index for changing the loaded map
        rem=j%3072
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        val_full.append(np.array(full))
        pssmth=np.load(data_path+'/smthps_'+str(i)+'_'+str(rem)+'.npy')
        val_pssmth.append(np.array(pssmth))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        val_label.append(np.array(label))
    
    #Up to this point: arrays loaded in val_(full,label,mfiltered).
        
    model4=load_model('cnn2_7min4ep')
    model8=load_model('cnn2_7min8ep')
    model12=load_model('cnn2_7min12ep')
    
    val_full=np.array(val_full).reshape(3072*nmaps-lenj,size,size,1)
    print('start')
    val_predictions4=model4.predict(val_full)
    val_predictions8=model8.predict(val_full)
    val_predictions12=model12.predict(val_full)
    print('end')

    return (val_predictions4, val_predictions8, val_predictions12 ,val_pssmth,val_full,val_label)

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


def total_comparator(val_predictions4, val_predictions8, val_predictions12, val_ps, val_full, val_label):
    """
    Compares the performance of the CNN and the MF. Change the internal parameters
    of the method if desired.
    """
    
    #above_detections,below_detections,undetected_above,undetected_below,real_objects,spur_objects)
    threshold4=[0.0248,0.0259] #CNN thresholds to be evaluated.
    threshold8=[0.069,0.082]
    threshold12=[0.08,0.11]
    num=50 #number of fluxes to be evaluated
    fluxes=np.linspace(1,500,num=num) #Flux in Tps
    criticalflux=600 #if a non-detection occurs above this Tps, prints the image.
    cnn4_completeness=np.zeros((len(threshold4),num))
    cnn4_spures=np.zeros((len(threshold4)))
    cnn8_completeness=np.zeros((len(threshold8),num))
    cnn8_spures=np.zeros((len(threshold8)))
    cnn12_completeness=np.zeros((len(threshold12),num))
    cnn12_spures=np.zeros((len(threshold12)))

    for thr_ind in range(len(threshold4)):
        thr=threshold4[thr_ind]
        above_detections=np.zeros((num))
        below_detections=np.zeros((num))
        undetected_above=np.zeros((num))
        undetected_below=np.zeros((num))
        real_objects=np.zeros((num))
        spur_objects=np.zeros((num))
        for ind in range(num):
            flux=fluxes[ind]
            for i in range(len(val_predictions4)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions4[i,:,:,0],val_ps[i],val_full[i,:,:,0],flux,val_label[i])
                
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                
                if flux>criticalflux and ua>0:
                    plt.figure()
                    plt.pcolormesh(val_predictions4[i,:,:,0])
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.pcolormesh(val_full[i,:,:,0])
                    plt.show()
                    plt.close()
                    print(i)
                    
        cnn4_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        cnn4_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    print('fin4')
    for thr_ind in range(len(threshold4)):
        thr=threshold8[thr_ind]
        above_detections=np.zeros((num))
        below_detections=np.zeros((num))
        undetected_above=np.zeros((num))
        undetected_below=np.zeros((num))
        real_objects=np.zeros((num))
        spur_objects=np.zeros((num))
        for ind in range(num):
            flux=fluxes[ind]
            for i in range(len(val_predictions8)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions8[i,:,:,0],val_ps[i],val_full[i,:,:,0],flux,val_label[i])
                
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                
                if flux>criticalflux and ua>0:
                    plt.figure()
                    plt.pcolormesh(val_predictions8[i,:,:,0])
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.pcolormesh(val_full[i,:,:,0])
                    plt.show()
                    plt.close()
                    print(i)
                    
        cnn8_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        cnn8_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    print('fin8')  
    for thr_ind in range(len(threshold12)):
        thr=threshold12[thr_ind]
        above_detections=np.zeros((num))
        below_detections=np.zeros((num))
        undetected_above=np.zeros((num))
        undetected_below=np.zeros((num))
        real_objects=np.zeros((num))
        spur_objects=np.zeros((num))
        for ind in range(num):
            flux=fluxes[ind]
            for i in range(len(val_predictions12)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions12[i,:,:,0],val_ps[i],val_full[i,:,:,0],flux,val_label[i])
                
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                
                if flux>criticalflux and ua>0:
                    plt.figure()
                    plt.pcolormesh(val_predictions12[i,:,:,0])
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.pcolormesh(val_full[i,:,:,0])
                    plt.show()
                    plt.close()
                    print(i)
                    
        cnn12_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        cnn12_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    
    
    #Comparison finished. Plot the basic results.
    plt.figure()
    colours=['red','blue','green','black']
    for thr_ind in range(len(threshold4)):
        thr=threshold4[thr_ind]
        plt.plot(fluxes*0.001920, cnn4_completeness[thr_ind,:], label='CNN4ep '+str(thr), color=colours[thr_ind], linestyle='dashed')
        print("Spureous ratio, 4 CNN at threshold "+str(thr)+": "+str(cnn4_spures[thr_ind]))
    for thr_ind in range(len(threshold8)):
        thr=threshold8[thr_ind]
        plt.plot(fluxes*0.001920, cnn8_completeness[thr_ind,:], label='CNN8ep '+str(thr), color=colours[thr_ind], linestyle='dotted')
        print("Spureous ratio, 8 CNN at threshold "+str(thr)+": "+str(cnn8_spures[thr_ind]))
    for thr_ind in range(len(threshold12)):
        thr=threshold12[thr_ind]
        plt.plot(fluxes*0.001920, cnn12_completeness[thr_ind,:], label='CNN12ep '+str(thr), color=colours[thr_ind], linestyle='solid')
        print("Spureous ratio, 12 CNN at threshold "+str(thr)+": "+str(cnn12_spures[thr_ind]))

    plt.xlabel('S / Jy')
    plt.ylabel('Completeness')
    plt.legend()
    plt.savefig('difftraincomp.svg',format='svg')
    plt.savefig('difftraincomp.png',format='png', dpi=300)
    plt.show()
    plt.close()
    return (above_detections,below_detections,real_objects,spur_objects,fluxes, cnn4_completeness, cnn4_spures, cnn8_completeness, cnn8_spures, cnn12_completeness, cnn12_spures)

#obtains a full comparison of the matched filter and the CNN
    
#val_predictions4,val_predictions8,val_predictions12,val_pssmth,val_full,val_label=image_loading(3072)

results2=total_comparator(val_predictions4, val_predictions8, val_predictions12 ,val_pssmth,val_full,val_label)

##SCRIPT FOR STUDYING SOURCES AND DETECTIONS. COMMENT IF NOT NEEDED.

cnn_detections_spur_list=np.array(cnn_detections_spur_list)
cnn_detections_true_list=np.array(cnn_detections_true_list)
#
##ORDER: Significancy, Ttotalmap, Tps


plt.figure(3)
#plt.title('Distribution of real detections CNN')
plt.scatter(cnn_detections_true_list[:,1], cnn_detections_true_list[:,0], s=3, color='green')
plt.scatter(cnn_detections_spur_list[:,1], cnn_detections_spur_list[:,0], s=4, color='red')
plt.xlim(-200,5200)
plt.xlabel(r'$T_{map}$ / $\mu$K')
plt.ylabel(r'CNN prediction')
#plt.savefig('cnndetectiondistr.png',format='png', dpi=300)
plt.show()
plt.close()
