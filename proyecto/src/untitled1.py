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

image = io.imread('../data/test/unknown/65.jpg')
image = color.rgb2gray(image)
kernel = np.array([[1,1,1],[1,4,1], [1,1,1]])
img_sharp = cv2.filter2D(image, -1, kernel)
imggauss = filters.gaussian(image, sigma=1)
img_256 = skimage.img_as_ubyte(image)


rho =1 #pixeles de distancia
theta = np.pi/180 #resolucion angular en radianes de la cuadricula de hough
threshold = 15 #minimo num de cortes en la cuadricula
min_line_length=50
max_line_gap = 10
line_image = np.copy(image)*0

plt.imshow(line_image, 'gray')

edges = np.uint8(np.array(image.shape))
edges = (feature.canny(imggauss, sigma=10)).astype(np.uint8)

#plt.imshow(edges, 'gray')
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0),5)
        
plt.imshow(line_image,'gray')





















