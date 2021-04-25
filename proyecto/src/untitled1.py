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
import matplotlib.pyplot as plt

image = cv2.imread('../data/test/unknown/43.jpg')
plt.imshow(image, 'gray')
imagegray = color.rgb2gray(image)
kernel = np.array([[1,1,1],[1,4,1], [1,1,1]])
img_sharp = cv2.filter2D(image, -1, kernel)
imggauss = cv2.GaussianBlur(imagegray, (5,5), 0)
img_256 = skimage.img_as_ubyte(image)


rho =1 #pixeles de distancia
theta = np.pi/180 #resolucion angular en radianes de la cuadricula de hough
threshold = 20 #minimo num de cortes en la cuadricula
min_line_length=50
max_line_gap = 4
line_image = np.copy(image)*0

'''
lower_white = np.array([0,0,255])
upper_white = np.array([255,255,255])
# Create the mask
mask = cv2.inRange(image, lower_white, upper_white)
 
# Create the inverted mask
mask_inv = cv2.bitwise_not(mask)

plt.imshow(mask_inv, 'gray')
'''

edges = np.uint8(np.array(image.shape))
edges = (feature.canny(imagegray, sigma=1.5)).astype(np.uint8)

plt.imshow(edges, 'gray')

lines = cv2.HoughLinesP(edges,1,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)

if lines is not None:
   for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image,(x1,y1), (x2,y2), (255,0,0),1)
        

plt.imshow(image, 'gray')
"""
lines = cv2.HoughLinesP(edges,1,theta,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)

if lines is not None:
    for i in range(0,len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2), (0,0,255),2)
        
plt.imshow(image)
"""
'''
plt.imshow(imggauss)

imgcanny = feature.canny(imggauss,1)

plt.imshow(imgcanny, 'gray')
'''


















