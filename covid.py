# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:14:45 2020

@author: Oscar
"""
import pandas as pd
#Cargar metadatos del dataset
df = pd.read_csv('C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/covid-chestxray-dataset-master/metadata.csv')
#De la columna 'finding' obtener todas las instancias con Covid
df_cov = df[df['finding'] =='COVID-19']

import glob 
import cv2
import numpy as np

#Covid
#Cargar imágenes de Covid del directorio, especificadas por el nombre que está en columna 'filename' de df_cov, en escala de grises
img_list = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob('C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/covid-chestxray-dataset-master/images/%s' % img) for img in df_cov['filename']] for item in i]
#Dimensiones
dim= (150,150)
#Ajustar todas las imágenes a 800x800
resCov = [cv2.resize(i, dim) for i in img_list]
#Convertir lista a array
resCov = np.asarray(resCov)
#Crear lista con label 0 = Covid
Covlab = [0] * len(resCov)
#x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

#Pneumonía
#Cargar imágenes de Neumonía del directorio, en escala de grises
img_list1 = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob('C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/chest_xray/test/PNEUMONIA/*')] for item in i]
#Borrar elementos para que sean la misma cantidad que Covid
del img_list1[len(resCov):]
#Ajustar las imágenes
resNeu = [cv2.resize(i, dim) for i in img_list1]
#Convertir lista a array
resNeu = np.asarray(resNeu)
#Crear lista con label 1 = Pneumonía
Neulab = [1] * len(resNeu)

#Normal
#Cargar imágenes de personas sanas del directorio, en escala de grises
img_list2 = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob('C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/chest_xray/test/NORMAL/*')] for item in i]
#Borrar elementos para que sean la misma cantidad que Covid
del img_list2[len(resCov):]
#Ajustar las imágenes
resNorm = [cv2.resize(i, dim) for i in img_list2]
#Convertir lista a array
resNorm = np.asarray(resNorm)
#Crear lista con label 2 = Normal
Normlab = [2] * len(resNorm)

#Concatenar los arrays de imágenes, orden= Cov, Pneu, Norm
fullArr = np.concatenate((resCov, resNeu, resNorm), axis=0)
#Añade una dimensión a las imágenes
fullArr = fullArr.reshape((fullArr.shape[0], fullArr.shape[1], fullArr.shape[2], 1))
#Convertir
maxm = np.amax(fullArr)
fullArr= fullArr.astype('float32')/maxm
#Concatenar las listas con labels, orden= Cov, Pneu, Norm
fullLab= Covlab+Neulab+Normlab

#Particionar (20%) y mezclar las instancias
from sklearn.model_selection import train_test_split
x_tr, x_tst, y_tr, y_tst= train_test_split(fullArr, fullLab, test_size=.20)

#Obtener las dimensiones de las imagenes
inputSh = x_tr.shape[1:] 

from keras import applications
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
#Crear instancia de ResNet50 (Model) porque agrupa las capas por objetos
#weights= None porque los inicializa aleatoriamente
#inc_top= false porque especificamos las dimensiones de entrada
base = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= inputSh)
#Modelo de la salida
x = base.output
#Disminuye el no de datos y prepara el modelo para la clasificación
x = GlobalAveragePooling2D()(x)
#50% de proba de que una neurona no se active
x = Dropout(0.5)(x)
#Capa de salida con las 3 clases, softmax para arrojar probabilidades
outpt = Dense(3, activation= 'softmax')(x)
#Pasar parámetros de entrada y salida al modelo para calcularlo 
model = Model(inputs = base.input, outputs = outpt)
#Compila el modelo con el optimizador adam, 
#loss sp_cat_cross porque las clases son mutuamente excluyentes y 
#accuracy porque es clasificación
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Ajusta el modelo con los train, 10 épocas y
#los analiza por grupos de 20 muestras
model.fit(x_tr, y_tr, epochs = 5, batch_size = 20, verbose=1)
#Evalúa y obtiene loss y accuracy con los tests
loss, acc = model.evaluate(x_tst, y_tst)
#Imprime el accuracy
print ('Test Accuracy = %.3f' % acc)
