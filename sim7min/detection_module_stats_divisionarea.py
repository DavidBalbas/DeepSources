# -*- coding: utf-8 -*-

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


def total_comparator(val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_label_weak, val_label_strong, val_pssmth_weak,val_pssmth_strong, positions_weak, positions_strong):
    """
    Compares the performance of the CNN and the MF. Change the internal parameters
    of the method if desired.
    """
    
    #above_detections,below_detections,undetected_above,undetected_below,real_objects,spur_objects)
    threshold=[0.124,0.150,0.180,0.230] #CNN thresholds to be evaluated.
    threshold2=[0.022] #MF thresh200olds to be evaluated (in sigmas)
    num=50 #number of fluxes to be evaluated
    fluxes=np.linspace(1,220,num=num) #Flux in Tps
    weak_completeness=np.zeros((len(threshold),num))
    weak_spures=np.zeros((len(threshold)))
    strong_completeness=np.zeros((len(threshold2),num))
    strong_spures=np.zeros((len(threshold2)))
    
    
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
            for i in range(len(val_predictions_weak)):
                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions_weak[i,:,:,0],val_pssmth_weak[i],val_full_weak[i,:,:,0],flux,val_label_weak[i])
                
                above_detections[ind]=above_detections[ind]+ad
                below_detections[ind]=below_detections[ind]+bd
                undetected_below[ind]=undetected_below[ind]+ub
                undetected_above[ind]=undetected_above[ind]+ua
                real_objects[ind]=real_objects[ind]+ro
                spur_objects[ind]=spur_objects[ind]+so
                    
        weak_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
        weak_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    
    fluxes2=fluxes
        
#    for thr_ind in range(len(threshold2)):
#        thr=threshold2[thr_ind]
#        above_detections=np.zeros((num))
#        below_detections=np.zeros((num))
#        undetected_above=np.zeros((num))
#        undetected_below=np.zeros((num))
#        real_objects=np.zeros((num))
#        spur_objects=np.zeros((num))
#        for ind in range(num):
#            flux=fluxes2[ind]
#            for i in range(len(val_predictions_strong)):
#                ad,bd,ua,ub,ro,so=img_accuracy_cnn_2(thr,val_predictions_strong[i,:,:,0],val_pssmth_strong[i], val_full_strong[i,:,:,0],flux,val_label_strong[i])
#                above_detections[ind]=above_detections[ind]+ad
#                below_detections[ind]=below_detections[ind]+bd
#                undetected_below[ind]=undetected_below[ind]+ub
#                undetected_above[ind]=undetected_above[ind]+ua
#                real_objects[ind]=real_objects[ind]+ro
#                spur_objects[ind]=spur_objects[ind]+so
#                
#        strong_completeness[thr_ind,:]=above_detections/(above_detections+undetected_above)
#        strong_spures[thr_ind]=spur_objects[0]/(spur_objects[0]+real_objects[0])
    
    #Comparison finished. Plot the basic results.
    plt.figure()
    colours=['red','blue','green','black']
    for thr_ind in range(len(threshold)):
        thr=threshold[thr_ind]
        plt.plot(fluxes*0.001920, weak_completeness[thr_ind,:], label='75-CNN '+str(thr), color=colours[thr_ind], linestyle='solid')
        print("Spureous ratio, weak CNN at threshold "+str(thr)+": "+str(weak_spures[thr_ind]))
#    for thr_ind in range(len(threshold2)):
#        thr=threshold2[thr_ind]
#        plt.plot(fluxes*0.001920, strong_completeness[thr_ind,:], label='75-CNN '+str(thr), color=colours[thr_ind])
#        print("Spureous ratio, strong CNN at threshold "+str(thr)+": "+str(strong_spures[thr_ind]))
    plt.xlabel('S / Jy')
    plt.ylabel('Completeness')
    plt.legend()
    plt.savefig('stats75weak-new.svg',format='svg')
#    plt.savefig('statsstrong75.png',format='png', dpi=300)
    plt.show()
    plt.close()
    return (above_detections,below_detections,real_objects,spur_objects,fluxes, weak_completeness, strong_completeness, weak_spures, strong_spures)

#obtains a full comparison of the matched filter and the CNN
    
val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_label_weak, val_label_strong, val_pssmth_weak,val_pssmth_strong, positions_weak, positions_strong=image_loading(3,int(0.75*3072))

results3=total_comparator(val_full_weak, val_full_strong, val_predictions_weak, val_predictions_strong, val_label_weak, val_label_strong, val_pssmth_weak,val_pssmth_strong, positions_weak, positions_strong)

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
plt.figure(3)
#plt.title('Distribution of real detections CNN')
##SWITCH ORDER TO CHANGE SUPERPOSITION OF RED AND GREEN
plt.scatter(cnn_detections_true_list[:,1], cnn_detections_true_list[:,0], s=3, color='green')
plt.scatter(cnn_detections_spur_list[:,1], cnn_detections_spur_list[:,0], s=4, color='red')
plt.xlim(-200,1500)
plt.xlabel(r'$T_{map}$ / $\mu$K')
plt.ylabel(r'CNN prediction')
plt.savefig('weakfore85detdistr.png',format='png', dpi=300)
plt.show()
plt.close()
