import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import glob
import os
import time
from tensorflow.keras import regularizers
import argparse
import sys
from math import *
from helper_func import *
from pathfile import *

vgg=get_vgg()
def test(imgfile):
    fastrcnn = get_fastrcnn()
    fastrcnn.load_weights(FASTRCNN_WEIGHTS)
    proposals=get_proposals(imgfile)
    fmapss=get_feature_map(imgfile,vgg)
    res=[]
    for b in proposals[0]:
        x,y,w,h=b/16.0
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)
        if w<=0 or h<=0:
            res.append([None])
            continue
        f=fmapss[y:y+h,x:x+w,:]
        inpimg=tf.image.resize(f,(14,14))
        inpimg=np.array(inpimg)
        inpimg=inpimg.reshape((1,14,14,512))
        res.append(fastrcnn.predict(inpimg))
    predictions=[]
    fboxes=[]
    for i in range(len(proposals[0])):
        if res[i]==[None]:
            continue
        x=int(proposals[0][i][0]+proposals[0][i][2]*res[i][0][0][0])
        y=int(proposals[0][i][1]+proposals[0][i][3]*res[i][0][0][1])
        w=int(proposals[0][i][2]*exp(res[i][0][0][2]))
        h=int(proposals[0][i][3]*exp(res[i][0][0][3]))
        tt=[x,y,w,h,res[i][1][0]]
        predictions.append(tt)
    predictions=sorted(predictions,key=lambda x:np.argmax(x[4]),reverse=True)
    bad=[]
    for i in range(len(predictions)):
        if i in bad:
            continue
        x,y,w,h=predictions[i][:4]
        for j in range(i+1,len(predictions)):
            x1,y1,w1,h1=predictions[j][:4]
            iou_val=iou([x,y,w+x,y+h],[x1,y1,x1+w1,y1+h1])
            if iou_val>=0.5:
                bad.append(j)
        fboxes.append(predictions[i])
    fboxes=sorted(fboxes,key=lambda x:max(x[4]),reverse=True)
    fboxes=fboxes[:3]

    im=cv2.imread(imgfile)
    im=cv2.resize(im,(800,800),interpolation=cv2.INTER_AREA)
    for temp in fboxes:
        if max(temp[4])<0.6:
            continue
        x,y,w,h=temp[:4]
        #print(x,y,w,h)
        if x+w>2500 or y+h>2500:
            continue
        print(temp)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
        text=str(temp[4].argmax())
        fh = open('code_to_name_mapping.txt', 'r')
        lines = fh.readlines()
        map = {}
        for line in lines:
            map[line.split()[0]] = line.split()[1]
        text = map[text]
    
        
        cv2.putText(im,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2_imshow(im)
    cv2.imwrite('annotated.jpg', im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagepath")
    parser.add_argument("--mode", help="'train' - if want to train, else - 'test'  without quotes ''")
    args = parser.parse_args()
    
    image_path = args.imagepath
    mode = args.mode

    if mode == 'test':
        test(image_path)
