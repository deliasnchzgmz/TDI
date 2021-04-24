
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% TDI PROYECTO 1: CLASIFICACIÓN DE IMÁGENES                             %
%                                                                       %
% Plantilla para implementar el sistema de clasificación de imágenes    %
% del proyecto.                                                         %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
import cv2
import skimage
import natsort
import numpy as np
#from skimage.filters.rank import entropy
from skimage import io, color, feature, measure #,filters, etc,
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import pandas as pd

def imageProcessing(image):

    """
    Esta función implementa la etapa de pre-procesado. El pre-procesado debe
    ser idéntico para todas las imágenes de train y test.

    La función recibe como entrada la imagen original ('image') a
    procesar. A la salida devuelve una variable de tipo
    diccionario ('processed_images') que contiene indexadas
    cada una de las imágenes resultantes de las diferentes
    técnicas de procesado escogidas (imagen filtrada,
    imagen de bordes, máscara de segmentación, etc.), utilizando para cada
    una de ellas una key que identifique su contenido
    (ej. 'image', 'mask', 'edges', etc.).

    Para más información sobre las variables de tipo
    diccionario en Python, puede consultar el siguiente enlace:
    https://www.w3schools.com/python/python_dictionaries.asp

    COMPLETE la función para obtener el conjunto de imágenes procesadas
    'processed_images' necesarias para la posterior extracción de características.
    Si lo necesita, puede definir y hacer uso de funciones auxiliares.
    """
    # El siguiente código implementa el BASELINE incluido en el challenge de
    # Kaggle.
    # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -

    processed_images = {}
    # Ejemplo: Añadimos la imagen original como una entrada a la variable diccionario
    processed_images["image"] = image
    # Añadimos la imagen en escala de grises como una entrada a la variable diccionario
    processed_images["image_gray"] = color.rgb2gray(image)
    # Añadimos la imagen en escala de grises con 256 niveles de gris para poder utilizarla en la matriz de co-ocurrencias
    processed_images["image_gray_256"] = skimage.img_as_ubyte(processed_images["image_gray"])
    #Añadimos la imagen tras aplicar un filtrado de sharpening
    kernel = np.array([[-1,-1,-1],[-1,4,-1], [-1,-1,-1]])
    processed_images["image_sharpening"] = cv2.filter2D(processed_images["image_gray"], -1, kernel)
    #Añadimos una imagen de bordes con canny a partid de image sharpening
    processed_images["image_bordes"] = feature.canny(processed_images["image_sharpening"], sigma=1)
    # Añadimos la imagen en escala de grises con 256 niveles de gris para poder utilizarla en la matriz de co-ocurrencias
    processed_images["image_gray_256"] = skimage.img_as_ubyte(processed_images["image_gray"])
    # Añadimos la mascara de la imagen como una entrada a la variable diccionario
    processed_images["image_binary"] = processed_images["image_sharpening"]
    # Añadimos la image en LAB
    image_lab = color.rgb2lab(color.gray2rgb(image))
    
    # Extraemos las componentes de la image_lab
    processed_images["image_lab_l"] = image_lab[:,:,0]
    processed_images["image_lab_a"] = image_lab[:,:,1]
    processed_images["image_lab_b"] = image_lab[:,:,2]
    
    
    # Añadimos el histograma de la imagen en escala de grises
    
    processed_images["image_histogram"] = np.histogram(processed_images["image_gray"], 256)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return processed_images

def extractFeatures(processed_images):

    """
    En esta función implementamos la etapa de extracción de características.

    La función recibe como entrada la variable 'processed_images'
    de tipo diccionario, la cual contiene las imágenes pre-procesadas
    obtenidas a partir de cada imagen de la base de datos. A la salida
    devuelve un vector con los valores de descriptores obtenidos
    para la imagen.

    Es posible que obtenga diferentes características para las diferentes
    versiones pre-procesadas de la imagen original. Concatene todas ellas para
    formar el vector final de características.

    Todas las imágenes deben ser representadas por vectores de características
    con el mismo número de features. Esta función aplica la misma lógica para
    las imágenes de todas las clases. También aplica la misma lógica para las
    imágenes de train y de test.

    """

    # El siguiente código implementa el BASELINE incluido en el challenge de
    # Kaggle.
    # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -
    features = []
    # Utilizamos la imagen en escala de grises para obtener, como descriptor
    # baseline, su desviación típica (descriptor muy simple de textura).
    image_gray = processed_images["image_gray"]
    std_gray = np.std(image_gray)
    features.append(std_gray)


    canny = np.sum(processed_images["image_bordes"]==1)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    features.append(canny)
    
    distances = [4] #Distancia entre los pares de pixeles que iremos acumulando la matriz de co-ocurrencias
  
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] #Array con los diferentes ángulos (en radianes) que nos indican la orientación a la hora de considerar un píxel vecino
  
    properties = ['contrast'] #El contraste sera la propiedad que hallemos a partir de la matriz de co-ocurrencias
  
    #Calculamos la matriz de co-ocurrencias normalizada a partir de los parametros anteriormente descritos
    glcm = feature.texture.greycomatrix(processed_images["image_gray_256"], distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
  
    #Calculamos el contraste para las cuatro combinaciones de pares de pixeles según su ángulo
    #Acumulamos en un array los 4 valores que pasaremos como caracteristicas al clasificador
    contrast = np.hstack([feature.texture.greycoprops(glcm, prop).ravel() for prop in properties]) 
    
    #Contamos el numero de objetos que hay en la imagen con regionprops()
    label_img = measure.label(processed_images["image_binary"])
    regions = measure.regionprops(label_img)
    nregions = len(regions)
    features.append(nregions)
    
    #Calculamos la transformada de fourier y diferentes caracteristicas
    fourier = np.fft.fft(processed_images["image_gray"])
    dep = np.abs(fourier) ** 2 #Densidad espectral de potencia
    fase = np.angle(fourier) #Angulo de fase
    mediaDep = np.mean(dep)
    mediaFase = np.mean(fase)
    desviacionDep = np.std(dep)
    desviacionFase = np.std(fase)
    #features.append(mediaFase)
    #features.append(mediaDep)
    #features.append(desviacionDep)
    features.append(desviacionFase)
    
    ##solas no - con las tres sale 76.25
    #features.append(np.mean(processed_images["image_lab_l"]))
    #features.append(np.mean(processed_images["image_lab_a"]))
    #features.append(np.mean(processed_images["image_lab_b"]))
    """
    desviacionHist = np.std(processed_images["image_histogram"])
    features.append(desviacionHist)
    
    """
    features = np.concatenate((features, contrast))
    
    return features

def databaseFeatures(db="../data/train"):

    """
    La función recibe como entrada una variable:
    - 'db', que es la ruta a la carpeta que contiene las
    imágenes de uno de los dos conjuntos de la base de datos: "train" o "test"

    A su salida, la función devuelve la matriz de características 'X' y
    una variable labels con las etiquetas (para el caso de train).

    ÚNICAMENTE ES NECESARIO MODIFICAR LA VARIABLE 'num_features' EN
    ESTA FUNCIÓN. 'num_features' es el número de descriptores que se están
    obteniendo para cada imagen.
    """

    folders = natsort.natsorted(os.listdir(db))
    imPaths = []
    labels  = []
    for i,f in enumerate(folders):

        numImages = len(os.listdir(db+'/'+f))
        imPaths.extend( [ db+'/'+f+'/'+str(i+1)+'.jpg' for i in range(numImages) ] )
        labels.extend( numImages*[(i+1)] )

    # Matriz de caracteristicas X
    # Para el BASELINE incluido en el challenge de Kaggle, se utiliza 1 feature
    num_features = 8 # MODIFICAR, INDICANDO EL NÚMERO DE CARACTERÍSTICAS EXTRAÍDAS
    num_images = len(imPaths)

    X = np.zeros( (num_images,num_features) )
    for i,imPath in enumerate(imPaths):

        #print("\rProcessing {}".format(imPath), end='')
        print('\r>>>> {}/{} done...'.format(i+1, len(imPaths)), end='')

        # leer la imagen del disco duro
        image = io.imread(imPath)

        # PREPROCESADO (ver función imageProcessing)
        processed_images = imageProcessing(image)

        # EXTRACCION DE CARACTERISTICAS (ver función extractFeatures)
        X[i,:] = extractFeatures(processed_images)

    print('')
    return X, np.array(labels)

def train_classifier(X_train, y_train, X_val = [], y_val = []):

    """
    Esta función no hace falta modificarla. Puede usted hacerlo si sabe lo que
    está haciendo aunque le recomendamos emplear su tiempo en diseñar mejores
    descriptores.

    La función recibe como entrada:

    - Las variables 'X_train', 'y_train', matriz de características
    y vector de etiquetas del conjunto de entrenamiento, respectivamente.
    Permiten obtenener un modelo de clasificación. Aunque son ustedes libres
    de modificar el clasificador, desaconsejamos esto y en su lugar, le
    proponemos que implemente mejores descriptores.

    - El vector de etiquetas 'y' contiene enteros del 1 al 4 que definen las
    diferentes categorías.

    - Las variables 'X_val', 'y_val', matriz de características y
    vector de etiquetas del conjunto de validación, respectivamente.
    Permiten validar los parámetros del algoritmo de clasificación escogido.
    Son variables opcionales. De nuevo, no le recomendamos utilizar estas
    variables sin estar seguro de lo que está haciendo.

    NOTA: Si se desea realizar un procedimiento de validación cruzada del
    modelo, puede subdividir el conjunto de entrenamiento ('X_train','y_train')
    en las particiones necesarias dentro de esta función. Muchos algoritmos de
    sklearn incluyen opciones para hacer esto internamente.

    A su salida, la función devuelve la variable 'scaler', para normalizar
    los datos de entrada, así como el modelo obtenido a partir
    del conjunto de imágenes de entrenamiento, 'model'.

    """

    # El siguiente código implementa el clasificador utilizado para el BASELINE
    # incluido en el challenge de Kaggle.
    # - - - NO ES NECESARIO MODIFICAR EL CLASIFICADOR PERO,
    # SI LO DESEA, PUEDE HACERLO Y SU PROPUESTA SE VALORARÁ POSITIVAMENTE - - -

    # Normalización ((X-media)/std >> media=0, std=1)
    scaler = StandardScaler()
    # Obtener estadísticos del conjunto de entrenamiento
    scaler.fit(X_train)
    # Normalización del conjunto de entrenamiento, aplicando los estadísticos.
    X_train = scaler.transform(X_train)

    # Definición y entrenamiento del modelo
    model = MLPClassifier(hidden_layer_sizes=(np.maximum(10,np.ceil(np.shape(X_train)[1]/2).astype('uint8')),
                                              np.maximum(5,np.ceil(np.shape(X_train)[1]/4).astype('uint8'))),
                          max_iter=200, alpha=1e-4, solver='sgd', verbose=0, random_state=1,
                          learning_rate_init=0.1)
    model.fit(X_train, y_train)
    
      
    ### - - - - - - - - - - - - - - - - - - - - - - - - -

    return scaler, model

def test_classifier(scaler, model, X_test):

    """
    Esta función no debe ser modificada. Se encarga de recibir el modelo de
    clasificación entrenado, el objecto 'scaler' y los datos de test, y
    produce las predicciones para las imágenes de test.

    La función recibe como entrada:

    - La variable 'scaler', que permite normalizar las características de
    las imágenes del conjunto de test de acuerdo con los estadísticos obtenidos
    a partir del conjunto de entrenamiento.
    - La variable 'model', que contiene el modelo de clasificación binaria obtenido
    mediante la función 'train_classifier' superior.
    - La variable 'X_test', matriz de características del conjunto de test, sobre
    la que se evaluará el modelo de clasificación obtenido a partir del conjunto
    de entrenamiento.

    A su salida, la función devuelve el vector de etiquetas predichas
    para las imágenes del conjunto de test.

    """

    # Normalización del conjunto de test, aplicando los estadísticos del
    # conjunto de entrenamiento
    X_test = scaler.transform(X_test)

    # Prediccciones sobre el conjunto de test
    y_pred = model.predict(X_test)
    ### - - - - - - - - - - - - - - - - - - - - - - - - -

    return y_pred

def eval_classifier(y, y_pred):

    """
    Esta función no debe ser modificada.

    La función recibe a su entrada el vector de etiquetas
    reales (Ground-Truth (GT)) 'y', así como el vector
    de etiquetas predichas 'y_pred' del conjunto de imágenes
    que se utiliza para evaluar las prestaciones del
    clasificador.

    A su salida, imprime la variable 'catAcc', con el valor
    obtenido para la medida de evaluación considerada en el challenge de kaggle
    (Categorization Accuracy).

    Aunque las etiquetas de test son desconocidas, es posible utilizar
    esta función con conjuntos de validación sacados del train o anotar
    manualmente el test. Si se decide evaluar utilizando una parte del conjunto
    de train, asegúrese de que el clasificador no recibe en entrenamiento las
    muestras que luego utilice para evaluar su comportamiento.
    """

    catAcc = np.sum(y == y_pred)/len(y)
    print('Categorization Accuracy: {:.2f}'.format(catAcc))

if( __name__ == '__main__'):

    """
    Cuando se ejecuta este script de python ('project1.py') la ejecución
    comenzará por aquí.

    Esta parte del código se encarga de realizar el proceso completo que
    produce el archivo .csv con las etiquetas estimadas para el conjunto de
    test que hay que subir a kaggle. Esto es:

        1) Se extraen las características de los conjuntos de train y test
            con la función 'databaseFeatures'.

        2) Se entrena el clasificador con los descriptores de las imágenes de
            de entrenamiento

        3) Se utiliza el clasificador para obtener las predicciones de las
            imágenes de test

        4) Se genera un archivo .csv con las predicciones para las imágenes
            de test que puede ser subido a kaggle como una 'submission'
    """

    dir_images_train_name='../data/train'
    dir_images_test_name='../data/test'
    submission_csv_name='../baseline.csv'

    # 1)
    print('Extrayendo descriptores de train y test...')
    X_train, y_train = databaseFeatures(db=dir_images_train_name)
    X_test, _ = databaseFeatures(db=dir_images_test_name)

    # 2)
    print('\nEntrenando el clasificador...')
    scaler, model = train_classifier(X_train, y_train)

    # 3)
    print('\nObteniendo predicciones para el conjunto de test...')
    y_pred = test_classifier(scaler, model, X_test)

    # 4)
    print('\ngenerando el archivo csv necesario para submission...')
    ids = [i+1 for i,e in enumerate(y_pred)]
    submission = pd.DataFrame({'Id':ids,'Category':y_pred})
    submission.to_csv(submission_csv_name,index=False)

    print('¡Listo!')
