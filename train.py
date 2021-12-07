import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from math import *
import glob
import os
import time
from tensorflow.keras import regularizers
from helper_func import *
import sys
import argparse
import random
from pathfile import *

vgg=get_vgg()
xtrain=[]
ytrain=[]
fh = open('classes.txt', 'r')
lines= fh.readlines()
classes = [line.stript() for line in lines]
fh.close()
for cat in classes:
    for file in glob.glob('Dataset/'+cat+'/images/*.jpg'):
        f=open('Dataset/'+cat+'/finalLabels/'+os.path.basename(file)[:-4]+'.txt')
        lines=f.readlines()
        orggt_boxes=[]
        f.close()
        for line in lines:
            temp=line.split()
            xmin,ymin,xmax,ymax=list(map(int,map(float,temp[1:])))
            orggt_boxes.append([xmin,ymin,xmax-xmin,ymax-ymin])
        tic=time.time()
        proposals=get_proposals(file)
        toc=time.time()
        #print('time to get boxes -',toc-tic)
        boxes=proposals[0]
        tic=time.time()
        fmaps=get_feature_map(file,vgg)
        toc=time.time()
        #print('time to get vgg feature map -',toc-tic)
        gt_boxes=get_resized_boxes(file,orggt_boxes)
        for box in boxes:
            test=None
            for gt in gt_boxes:
                newbox=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
                if iou(gt,newbox)>=0.5:
                    test=gt
            if test == None:
                continue
            tx=(test[0]-box[0])/box[2]
            ty=(test[1]-box[1])/box[3]
            tw=log(test[2]/box[2])
            th=log(test[3]/box[3])
            x,y,w,h=box/16.0
            x=int(x)
            y=int(y)
            w=int(w)
            h=int(h)
            fmap=fmaps[y:y+h,x:x+w,:]
            try:
                inpimg=tf.image.resize(fmap,(14,14))
            except:
                continue
            
            fh = open('code_to_name_mapping.txt', 'r')
            lines = fh.readlines()
            fh.close()

            for line in lines:
              if line.split()[1] == cat:
                code = int(line.split()[0])
                break
            cl = [0.0]*len(classes)
            cl[code] = 1.0

            t_arr=np.array([tx,ty,tw,th])
            cl=np.array(cl)
            inpimg=np.array(inpimg)
            xtrain.append(inpimg)
            ytrain.append([t_arr,cl])
            #model.fit(x=inpimg,y=(t_arr,cl))
temp1=[]
temp2=[]
for t in ytrain:
    temp1.append(t[0])
    temp2.append(t[1])
temp1=np.array(temp1)
temp2=np.array(temp2)
xtrain=np.array(xtrain)


for _ in range(1):
    p=random.sample(list(zip(xtrain,temp1,temp2)),1200)
    tempx,tempy1,tempy2=zip(*p)
    tempx=np.array(tempx)
    tempy1=np.array(tempy1)
    tempy2=np.array(tempy2)
    fastrcnn = get_fastrcnn()
    hist=fastrcnn.fit(x=tempx,y=[tempy1,tempy2],batch_size=500,epochs=45)
    fastrcnn.save_weights(FASTRCNN_WEIGHTS)
