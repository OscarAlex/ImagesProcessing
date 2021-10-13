# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:59:39 2020

@author: Oscar
"""
import tensorflow as tf
import pathlib

import cv2 
import os 
import glob 
import numpy as np 

#%% Links
PATH_A= 'C:/Users/Oscar/Documents/Python Scripts/PPIA/rocks/Loma Linda/Diorite'
PATH_B= 'C:/Users/Oscar/Documents/Python Scripts/PPIA/rocks/Loma Linda/Gabbro'
PATH_C= 'C:/Users/Oscar/Documents/Python Scripts/PPIA/rocks/Loma Linda/Granite'
PATH_D= 'C:/Users/Oscar/Documents/Python Scripts/PPIA/rocks/Loma Linda/Granodiorite'

#Método para obtener las imágenes
def getImas(path, lab):
    data_path = os.path.join(path,'*g')
    data= [cv2.imread(f1) for f1 in glob.glob(data_path)]
    #data= [cv2.imread(f1, cv2.IMREAD_GRAYSCALE) for f1 in glob.glob(data_path)]
    #Ajustar todas las imágenes
    data = [cv2.resize(i, (100,100)) for i in data]
    #Convertir a array
    data= np.array(data)
    siz = [lab] * len(data)
    print(lab, " ", len(siz))
    return data, siz

#Obtener las imagenes
A, ALab= getImas(PATH_A, 0)
B, BLab= getImas(PATH_B, 1)
C, CLab= getImas(PATH_C, 2)
D, DLab= getImas(PATH_D, 3)

#%% Normalizar y unir
#Concatenar los arrays de imágenes
fullArr = np.concatenate((A, B, C, D), axis=0)
#Normalizar
fullArr= fullArr.astype('float32')/255
#Concatenar las listas con labels, orden= Cov, Pneu, Norm
fullLab= ALab+BLab+CLab+DLab
#Imprimir dimensiones
print("Features =", fullArr.shape)
print("Labels =", len(fullLab))

#Particionar y mezclar las instancias
fullLab = np.array(fullLab)

#%% Modelo
#Capas
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.ZeroPadding2D(padding=(2,2), input_shape= (100,100,3)))
model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape= (100,100,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% Entrenar el modelo
model.fit(fullArr, fullLab, epochs=10, batch_size=32, verbose=1)

#model.summary()
#%% Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

#Guardar modelo
tflite_model_file = pathlib.Path('C:/Users/Oscar/Documents/Python Scripts/PPIA/rocks/rock_mdl_new.tflite')
tflite_model_file.write_bytes(tflite_model)
