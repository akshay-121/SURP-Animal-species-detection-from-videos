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
from pathfile import *

def iou(box_a,box_b):
    y1=max(box_a[1],box_b[1])
    y2=min(box_a[1]+box_a[3],box_b[1]+box_b[3])
    x1=max(box_a[0],box_b[0])
    x2=min(box_a[0]+box_a[2],box_b[0]+box_b[2])
    if y1>y2 or x1>x2:
        return 0                                                    #iou calculation
    inter_area=(x2-x1)*(y2-y1)
    union_area=box_a[2]*box_a[3]+box_b[2]*box_b[3]-inter_area
    if union_area == 0:
        return 0
    return inter_area/union_area
def get_proposals(path):
    im=cv2.imread(path)
    im=cv2.resize(im,(800,800),cv2.INTER_AREA)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(MODEL_YML)
    edges = edge_detection.detectEdges(np.float32(im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(64)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    return boxes
def get_feature_map(path,model):
    im=tf.keras.preprocessing.image.load_img(path=path,target_size=(800,800))
    im = tf.keras.preprocessing.image.img_to_array(im)
    im=tf.keras.applications.vgg16.preprocess_input(im)
    im=np.reshape(im,(1,800,800,3))
    pred=model.predict(im)[0]
    return pred
def get_vgg():
    m=VGG16(include_top=False,input_shape=(800,800,3))
    x=Input((800,800,3))
    inp=x
    for layer in m.layers:
        if layer.__class__.__name__=='InputLayer':
            continue
        if layer.output_shape[1]>=50:
            x=layer(x)
    model=Model(inp,x)
    model.trainable=False
    return model
def get_resized_boxes(path,original_boxes):
    temp=cv2.imread(path)
    gt_boxes=[]
    width,height=len(temp[0]),len(temp)
    temp=cv2.resize(temp,(800,800),interpolation=cv2.INTER_AREA)
    for gt in original_boxes:
        x,y,w,h=list(map(int,gt))
        x=int(x*(800.0/width))
        y=int(y*(800.0/height))
        w=int(w*(800.0/width))
        h=int(h*(800.0/height))
        #print(x,y,w,h)
        gt_boxes.append([x,y,w,h])
    return gt_boxes
def get_fastrcnn():
    y=Input((14,14,512))
    y_inp=y
    y=MaxPool2D(2)(y)
    y=Flatten()(y)
    y=Dense(1024)(y)
    y=Dropout(0.25)(y)
    y=Dense(1024)(y)
    y=Dropout(0.25)(y)
    y=Dense(512)(y)
    y=Dropout(0.25)(y)
    y=Dense(1024,name='logits')(y)
    reg=Dense(4,activity_regularizer=regularizers.l2(1e-1),name='regression_layer')(y)
    cls=Dense(11,name='class_layer')(y)
    cls=Softmax()(cls)
    fastrcnn=Model(inputs=y_inp,outputs=[reg,cls])
    return fastrcnn