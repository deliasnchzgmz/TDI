# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import skimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import io


#EJERCICIO 1
x=io.imread('dreamers.png') #leemos la imagen dreamers.png
plt.imshow(x)  #la imagen se plotea con unos colores falsos
#plt.show()

#nos pide que visualicemos la imagen con 8 niveles de gris
#para ello especificamos el mapa de color en escala de grises con cmap='gray'

gray = LinearSegmentedColormap.from_list('eightmap', [(0,0,0), (1,1,1)], N=8)
#he creado un mapa de color que va desde el blanco al negro con 8 niveles de gris
#ploteo la imagen usando el mapa de color creado anteriormente
plt.imshow(x, cmap=gray)
plt.show()

#ploteo el mapa de color en escala de grises
m=np.array([np.arange(200) for i in range(200)])
plt.imshow(np.transpose(m), cmap=gray)
plt.show()

#creamos un nuevo espacio de color esta vez tomando como colores extremo las esquinas del cubo

rgb = LinearSegmentedColormap.from_list('cuboRGB', [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)], N=8)

#ploteo la imagen usando el mapa de color creado anteriormente
plt.imshow(x, cmap=rgb)
plt.show()

#ploteamos este mapa de color
m=np.array([np.arange(200) for i in range(200)])
plt.imshow(np.transpose(m), cmap=rgb)
plt.show()

#nos pide que consideremos cuál de los mapas es el más indicado para el ejemplo de las imágenes infrarrojas
#el más adecuado es el llamado jet, en miscelánea
rtob = LinearSegmentedColormap.from_list('rtob', [(0,0,1),(1,0,0)], N=256)
plt.imshow(x, cmap='jet') #busco en colormaps
plt.show()

#EJERCICIO 2

#cargamos la imagen peppers.png en una matriz de tres dimensiones
#la imagen se carga directamente en una matriz 3D porque está en color
p = io.imread('peppers.png')

#extraemos la componente roja de la imagen, seleccionando solo 0 y la ploteo en escala de grises
red_c = p[:,:,0]
plt.imshow(red_c, cmap=gray)
#en pseudo color (sin parámetro cmap)
plt.imshow(red_c)

#en rojo
plt.imshow(red_c, LinearSegmentedColormap.from_list('eightmap', [(1,0,0), (1,1,1)], N=8))

#EJERCICIO 3
#el mapa de colores que necesitamos es el anteriormente creado rgb

black_square = np.zeros([1,1,3], dtype=np.uint8)
black_square[:]=0
#plt.imshow(black_square, rgb)
red_square = np.zeros([1,1,3])
red_square[:,:,:]=[1,0,0]
#plt.imshow(red_square, rgb)
blue_square = np.zeros([1,1,3])
blue_square[:,:,:]=[0,0,1]
#plt.imshow(blue_square, rgb)
green_square = np.zeros([1,1,3])
green_square[:,:,:]=[0,1,0]
#plt.imshow(green_square, rgb)
yellow_square = np.zeros([1,1,3])
yellow_square[:,:,:]=[1,1,0]
#plt.imshow(yellow_square, rgb)
magenta_square = np.zeros([1,1,3])
magenta_square[:,:,:]=[1,0,1]
#plt.imshow(magenta_square, rgb)
cyan_square = np.zeros([1,1,3])
cyan_square[:,:,:]=[0,1,1]
#plt.imshow(cyan_square, rgb)
white_square = np.zeros([1,1,3])
white_square[:]=[1]
#plt.imshow(white_square, rgb)

#concatenamos horizontalmente (axis 1) los cuadrados
first_row = np.concatenate((black_square, red_square, blue_square), axis=1)
second_row = np.concatenate((green_square, yellow_square, magenta_square), axis=1)
third_row = np.concatenate((cyan_square, white_square, black_square), axis=1)
#ahora concatenamos verticalmente(axis=0) las filas
rgb_square = np.concatenate((first_row, second_row, third_row), axis=0)
plt.imshow(rgb_square, rgb) #mosaico de colores

#EJERCICIO 4
#el rojo en hsv es (0,1,1)
hsv = np.zeros([200, 200, 3])
hsv[:,:,0] = 0  #componente h (hue) es 0
hsv[:,:,1] = 1  #componente s (saturation) es 1
hsv[:,:,2] = 1  #componente v (value) es 1

rgb_r = skimage.color.hsv2rgb(hsv) #paso la imagen hsv a rgb

plt.imshow(rgb_r, rgb)  #ploteo la imagen en rgb


#EJERCICIO 5
img = io.imread('peppers.png')
r_img = img[:,:,0]  #componente roja
g_img = img[:,:,1]  #componente verde
b_img = img[:,:,2]  #componente azul

#las cargo en la misma figura
fig, (ax1,ax2,ax3) = plt.subplots(3,1) #subplot en vertical
fig.suptitle('Componentes RGB')

ax1.imshow(r_img, cmap='gray')
ax2.imshow(g_img, cmap='gray')
ax3.imshow(b_img, cmap='gray')


#EJERCICIO 6
img1 = io.imread('SL-V.bmp')
img2 = io.imread('SL-IR.bmp')

fig, (ax1,ax2) = plt.subplots(1,2)  #subplot en horizontal
fig.suptitle('Imágenes visible e infrarroja')

ax1.imshow(img1)
ax2.imshow(img2)


#me piden que convierta la visible a escala de grises
#elimino componentes a color
gris_img1 = skimage.color.rgb2gray(img1)
#plt.imshow(gris_img1)
fig, (ax1,ax2) = plt.subplots(1,2)  #subplot en horizontal
fig.suptitle('Imagen visible en escala de grises y en pseudocolor')

#ploteo en escala de grises y en pseudocolor, se aprecia mucho mejor en escala de grises
ax1.imshow(gris_img1, gray)
ax2.imshow(gris_img1)

#separo las componentes para cada una de las imágenes
#primero de la imagen visible
r_img1 = img1[:,:,0]  #componente roja
g_img1 = img1[:,:,1]  #componente verde
b_img1 = img1[:,:,2]  #componente azul
#ahora de la imagen infrarroja
r_img2 = img2[:,:,0]  #componente roja
g_img2 = img2[:,:,1]  #componente verde
b_img2 = img2[:,:,2]  #componente azul

#ahora las agrupo en una misma figura
fig, ax = plt.subplots(2,3)  #subplot en horizontal

fig.suptitle('Componentes RGB de visible e infrarroja')
ax[0,0].imshow(r_img1, gray)
ax[0,1].imshow(g_img1, gray)
ax[0,2].imshow(g_img1, gray)

ax[1,0].imshow(r_img2, gray)
ax[1,1].imshow(g_img2, gray)
ax[1,2].imshow(b_img2, gray)

#sustituyo una de las compnentes de la imagen visible (G) por una de las componentes infrarrojas
#??????????????????????

#fig, ax = plt.subplots(2,3)  #subplot en horizontal

#fig.suptitle('Componentes RGB de visible e infrarroja')
#ax[0,0].imshow(r_img1, gray)
#ax[0,1].imshow(g_img2, gray)
#ax[0,2].imshow(g_img1, gray)

#plt.imshow()
