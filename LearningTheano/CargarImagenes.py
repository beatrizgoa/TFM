# Se va a definir una funcion en la que se cargan imagenes y se guarden para luego pasarlas a la red de conv.
#It is defined a fuction in which images are loaded and saved in order to use in a CNN

#Se parte de que las imagenes son todas iguales de tamanyo
#It is supposed that all the images have the same size/shape.

import cv2
import os
import Image
import numpy as np


def CargarIm(in_path):
    image_list=[]

    imagenes = os.listdir(in_path)


    #Leemos cada una de las imagenes
    for posicion, imagen in enumerate(imagenes):

        im= Image.open(in_path + '/' + imagen)
        im=np.array(im)
        image_list.append(im)

    return image_list


def main():

    in_path='/home/beaa/Escritorio/theano/CargarImagenes'

    image_list=CargarIm(in_path)

    tam=len(image_list)

    print ('lista')
    print image_list

    print ("Tamanyo lista")
    print tam