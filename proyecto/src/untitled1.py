# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:00:48 2021

@author: dl2pa
"""
import os
import cv2
import skimage
import natsort
import numpy as np
from scipy.stats import entropy
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage import io, color, feature, measure ,filters, exposure
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import pandas as pd
import math
import matplotlib.pyplot as plt


image = cv2.imread('../data/test/unknown/2.jpg')
#plt.imshow(image, 'gray')
dsize = (300, 300)
image = cv2.resize(image, dsize )
imagegray = color.rgb2gray(image)

#imggauss = cv2.medianBlur(imagegray, 11, 0)

kernel = np.array([[-1/9,-1/9,-1/9],[-1/9,(5-1/9),-1/9], [-1/9,-1/9,-1/9]])
imgcontrast = exposure.equalize_hist(imagegray)
img_sharp = cv2.filter2D(imgcontrast, -1, kernel)
img_256 = skimage.img_as_ubyte(image)
edges = np.uint8(np.array(image.shape))
edges = (feature.canny(imagegray, sigma=2)).astype(np.uint8)
img_canny1 = feature.canny(imagegray, sigma=3).astype(np.uint8)
img_canny2 = feature.canny(imagegray, sigma=4).astype(np.uint8)
canny = 3*img_canny1 - img_canny2


y = np.arange(80).reshape(80,1)
for i in range(20):
    y[i] = 1
for i in range(20, 40, 1):
    y[i] = 2
for i in range(40, 60, 1):
    y[i] = 3
for i in range(60, 80, 1):
    y[i] = 4
    
# dsize


mask = (img_sharp > filters.threshold_mean(img_sharp)).astype(np.uint8)
sum = np.mean(mask)

blur1 = cv2.medianBlur(mask, 19)

th3 = cv2.adaptiveThreshold(imgcontrast.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

label_img = measure.label(blur1)
regions = measure.regionprops(label_img)
nregions = len(regions)

#plt.imshow(th3, 'gray')
plt.show()

#plt.subplot(1,2,1)
#plt.imshow(mask, 'gray')
plt.show()
#plt.imshow(blur1, 'gray')
plt.show()
#plt.subplot(1,2,2)
#plt.imshow(resize)

grad = np.array([[0,1,0],[1,-8,1], [0,1,0]])
grad1 = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
imggrad = cv2.filter2D(imgcontrast, -1, grad1)
cn =  feature.canny(imggrad, sigma=4).astype(np.uint8)
plt.imshow(imgcontrast, 'gray')
plt.show()
#plt.imshow(imggrad, 'gray')
#plt.show()
#plt.imshow(cn, 'gray')
plt.show()

line_image = np.copy(imggrad)*0

theta = np.pi/180 #resolucion angular en radianes de la cuadricula de hough
threshold = 40 #minimo num de cortes en la cuadricula
min_line_length=50
max_line_gap = 10
count = 0
lines = cv2.HoughLinesP(cn.astype(np.uint8),1,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
if lines is not None:
       for line in lines:
        count = count+1
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0),1)     
       X1 = lines[:,0,0]
       X2 = lines[:,0,1]
       Y1 = lines[:,0,2]
       Y2 = lines[:,0,3]      
       
       slope = (Y2-Y1)/(X2-X1)
       stdSlope = np.std(slope)
       meanSlope = np.mean(slope)
       meanLength = np.mean(((X2-X1)**2+(Y2-Y1)**2)**0.5)
       stdLength = np.std(((X2-X1)**2+(Y2-Y1)**2)**0.5)
       

#plt.imshow(imggrad,'gray')
plt.show()

imggrad2 = cv2.filter2D(imggrad, -1, grad1)

plt.imshow(line_image,'gray')
plt.show()
'''
#5
rho =1 #pixeles de distancia
theta = np.pi/180 #resolucion angular en radianes de la cuadricula de hough
threshold = 50 #minimo num de cortes en la cuadricula
min_line_length=100
max_line_gap = 10
line_image = np.copy(image)*0


#pt.imshow(edges, 'gray')

edges = feature.canny(img_sharp, sigma=1.5)
plt.imshow(edges, 'gray')
lines = cv2.HoughLinesP(edges,1,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
if lines is not None:
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.lines(line_image,(x1,y1), (x2,y2), (255,0,0),5)
        
plt.imshow(lines)
'''  
'''
X1 = lines[:,0,0]
X2 = lines[:,0,1]
Y1 = lines[:,0,2]
Y2 = lines[:,0,3]

a = (X2-X1)
b = (Y2-Y1)
c = np.mean(((X2-X1)**2+(Y2-Y1)**2)**0.5)



print(lines)

#plt.imshow(image, 'gray')
'''

'''

resize para matriz de coocurrencias

para la clase partitura una buena es lo de los colores(en teoria porque luego como hay colores pues no es tan buena)

#plot train
    feat = X_train[:,0]
    plt.figure()
    plt.subplot(1,4,1)
    plt.plot(feat[np.where(y_train==1), color='red'])
    plt.figure()
    plt.subplot(1,4,2)
    plt.plot(feat[np.where(y_train==2), color='green'])
    plt.figure()
    plt.subplot(1,4,3)
    plt.plot(feat[np.where(y_train==3), color='blue'])
    plt.figure()
    plt.subplot(1,4,4)
    plt.plot(feat[np.where(y_train==4), color='black'])
 #buena caracteristica para diferenciar brain y sheetmusic de helechos y grapes
 luego si hacemos una que diferencie brains y sheetmusic ya se diferecniaria

a plt.plot se le puede meter un sort np.sort() antes del feat de plt.plot   

podemos usar el train  y test escaladas que es lo primero que se hace al entrar

se puede utilizas scatter

para hough quedarnos con las lineas grandes (grande r) o que sean mas o menos paralelas



pca (seleccionar caracteristicas, lo explican el miercoles)
'''

















