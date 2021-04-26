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
from skimage import io, color, feature, measure ,filters
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import pandas as pd
import math
import matplotlib.pyplot as plt

image = cv2.imread('../data/test/unknown/58.jpg')
plt.imshow(image, 'gray')
imagegray = color.rgb2gray(image)
imggauss = cv2.GaussianBlur(imagegray, (5,5), 0)
kernel = np.array([[-1/9,-1/9,-1/9],[-1/9,5,-1/9], [-1/9,-1/9,-1/9]])
img_sharp = cv2.filter2D(imagegray, -1, kernel)
img_256 = skimage.img_as_ubyte(image)

plt.subplot(1,2,1)
plt.imshow(imagegray, 'gray')
plt.subplot(1,2,2)
plt.imshow(img_sharp, 'gray')
'''
#5
rho =1 #pixeles de distancia
theta = np.pi/180 #resolucion angular en radianes de la cuadricula de hough
threshold = 50 #minimo num de cortes en la cuadricula
min_line_length=40
max_line_gap = 10
line_image = np.copy(image)*0

line_image = np.copy(image)*0


edges = np.uint8(np.array(image.shape))
edges = (feature.canny(imagegray, sigma=2)).astype(np.uint8)

plt.imshow(edges, 'gray')
count = 0

edges = np.uint8(np.array(image.shape))
edges = (feature.canny(imagegray, sigma=1.5)).astype(np.uint8)
plt.imshow(edges, 'gray')
lines = cv2.HoughLinesP(edges,1,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
if lines is not None:
   for line in lines:
    count = count+1
    for x1,y1,x2,y2 in line:
        cv2.line(image,(x1,y1), (x2,y2), (255,0,0),1)
        
        
X1 = lines[:,0,0]
X2 = lines[:,0,1]
Y1 = lines[:,0,2]
Y2 = lines[:,0,3]

a = (X2-X1)
b = (Y2-Y1)
c = np.mean(((X2-X1)**2+(Y2-Y1)**2)**0.5)


#plt.imshow(image, 'gray')

'''
'''
resize para matriz de coocurrencias
'''

















