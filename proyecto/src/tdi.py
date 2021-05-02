# -*- coding: utf-8 -*-
"""tdi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17Esmb9r1wpIhkdUAChc9pEwlNZO54IF3
"""

import os, math, cv2, skimage, natsort
import mahotas
import numpy as np
import pandas as pd
#import tensorflow as tf
from scipy.stats import entropy
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage import io, color, feature, measure ,filters, exposure
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
scaler = StandardScaler()

base_dir = '../data'
submission_csv_name='../delia.csv'

train_dir = os.path.join(base_dir, 'train')
#validation_dir = os.path.join(base_dir, 'test')
test_dir = os.path.join(base_dir, 'test')


# Directory with our training cat/dog pictures
train_brain_dir = os.path.join(train_dir, 'brain')
train_fern_dir = os.path.join(train_dir, 'fern')
train_grapes_dir = os.path.join(train_dir, 'grapes')
train_music_dir = os.path.join(train_dir, 'sheet-music')

# Directory with our validation cat/dog pictures
#validation_cats_dir = os.path.join(validation_dir, 'cats')
#validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print('total training brain images :', len(os.listdir(      train_brain_dir ) ))
print('total training fern images :', len(os.listdir(      train_fern_dir ) ))
print('total training grapes images :', len(os.listdir(      train_grapes_dir ) ))
print('total training music images :', len(os.listdir(      train_music_dir ) ))

import tensorflow as tf

model = tf.keras.models.Sequential([
    # el input es de 150x150 en RGB
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Siempre terminamos una "flatten"
    tf.keras.layers.Flatten(),
    # 512 neurones en una capa densa "hidden"
    tf.keras.layers.Dense(512, activation='relu'), 
    # El output es una sola neurona. 1 para la clase ('cats') y cero para la clase ('dogs')
    tf.keras.layers.Dense(4, activation='softmax')  
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Es siempre recomendable normalizar las imágenes (escalar)
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

# --------------------
# Usaremos un batch de 20 para entrenar
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    target_size=(150, 150))
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  batch_size=20,
                                                  target_size=(150,150))

history = model.fit(train_generator,
                    steps_per_epoch=15,
                    epochs=20,
                    validation_steps=50,
                    verbose=2)

#scaler.fit(test_generator)
#test_generator = scaler.transform(test_generator)

    # Prediccciones sobre el conjunto de test
y_pred = np.hstack(model.predict(test_generator))

ids = [i+1 for i,e in enumerate(y_pred)]
submission = pd.DataFrame({'Id':ids,'Category':y_pred})
submission.to_csv(submission_csv_name,index=False)
