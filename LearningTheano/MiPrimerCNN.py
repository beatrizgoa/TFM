

#Imports
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy as np
import gzip
import cPickle


def LoadDatabase():
    data=[]
    dataset='mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    data[0]=train_set
    data[1]=valid_set
    data[2]=test_set

    return data

# Se va a comenzar definiendo las capas
class ConvPool:
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        #filter shape=(number of filters, num input feature maps,filter height, filter width)
        #image_shape: (batch size, num input feature maps,image height, image width)

        self.input=input

        self.w=theano.shared(np.asarray(rng.uniform(size=filter_shape,dtype=theano.config.floatX),borrow=True))



def main ():

    #Se cargan las cariables.
    data=LoadDatabase()
    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x, test_set_y = data[2]

    #Se crea el aleatorio para los pesos
    rng = np.random.RandomState(23455)

    #Numero de elementos
    tam_lote=500

    #Se definen entradas y salidas:
    x=T.matrix('x')
    y=T.vector('y',dtype=int)


    #La capa cero sera la capa de entrada de las imagens que son de tamaño 28x28.
    #La  capa cero corresponde a las imagenes de entrada por lo qye hay que hacer que la entrada x sea como las imagenes

    Capa0 = x.reshape((tam_lote, 1, 28, 28))

    #El tamalo del filtro sera el numero del lote, 1 caracteristica y el tamaño de las imagenes

    PrimeraCapa=ConvPool(rng,Capa0,filter_shape=(tam_lote,4,28,28), image_shape= (tam_lote,1,28,28))



if __name__ == '__main__':
    main()



