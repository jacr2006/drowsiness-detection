# importacion de los paquetes
from imutils.video import VideoStream
import argparse
import time
import cv2
import time
import numpy as np

import keras
from keras.models import load_model
import tensorflow as tf

# Para el "deployment" del modelo  en la pc o en una app, debe disponerse de la misma version de "keras" usada en el entrenamiento
print(tf.__version__)
print(keras.__version__)

#parametros de la imagen
IMG_WIDTH = 96# tamaño de a imagen de entrada de la CNN
IMG_HEIGHT = 96
IMG_CHANNEL = 1# imagen en escala de grises
#N_LANDMARK = 68# numero de marcas faciales
N_LANDMARK = 12# numero de marcas faciales

#parametros de normalizacion
MEAN = 38.47239423802548# valor medio de la normalizacion de las marcas faciales
RANGE = 96.0#rango de la normalizacion de las marcas faciales

# argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

# carga el modelo para deteccion de rostros: haarcascade
detector = cv2.CascadeClassifier(args["cascade"])
# carga del modelo para la deteccion de las marcas faciales de los ojos
model = load_model( args["model"] )
    
# inicio de la captura del video
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 
while True:
    # captura de un frame
    frame = vs.read()
    
    # conversion de la imagen de entrada
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # deteccion del rostro
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if (len(rects)>0):
        x = rects[0][0]#cooordenadas de la deteccion
        y = rects[0][1]
        w = rects[0][2]
        h = rects[0][3]
        
        #preparacion de la imagen de entrada: color, tamaño y normalizacion
        grayImg = gray[y:(y+h), x:(x+w)]#extrae rectangulo del rostro detectado en escala de grises
        rezImg = cv2.resize( grayImg, ( IMG_WIDTH, IMG_HEIGHT ) )# escalado de la imagen
        rshImg = rezImg.reshape(1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)#reordenamiento de la imagen de acuerdo a la entrada requerida por la CNN
        rshImg = rshImg/255.0#normalizacion de la imagen, entre (0,1). De acuerdo al entrenamiento

        # inferencia en el modelo
        start = time.time()    
        res=model.predict(rshImg)# ejecucion de la inferencia
        print("tiempo de la inferencia: {}".format(time.time() - start))    
        
        #preparacion del resultado para el despliegue
        marks=res.reshape( 1, N_LANDMARK, 2 )
        marks = ( marks * RANGE ) + MEAN #desnormalizacion de la inferencia       
        
        #despliegue del resultado
        for i in range(N_LANDMARK):#dibuja la inferencia sobre la imagen, escalado (tamaño de la inferencia->tamaño de la deteccion)
            cv2.circle( grayImg, (  int( marks[0,i,0]*grayImg.shape[0]/IMG_WIDTH ), 
                                                int( marks[0,i,1]*grayImg.shape[1]/IMG_HEIGHT )  ), 
                                                2, (255, 0, 0), -1 )

        for i in range(h):# copia el resultado sobre el frame de salida
            for j in range(w):
               gray[i,j]= grayImg[i,j]
                
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)              
        
    # despliegue de la imagen con el resultado
    cv2.imshow("Frame", gray)
    key = cv2.waitKey(1) & 0xFF

    #
    if key == ord("q"):
        break

# 
cv2.destroyAllWindows()
vs.stop()
