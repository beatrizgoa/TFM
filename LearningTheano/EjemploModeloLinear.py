import theano
from theano import tensor as T
import numpy as np

#Se declarann las variables
X=T.scalar()
Y=T.scalar()

#Se construye el modelo
def model(X,W):
    return X*W

#Se crean los pesos
W=theano.shared(np.asarray(0.,dtype=theano.config.floatX))
#Aqui W se inicializa a 0


#Se le pasa al modelo la entrada y los pesos
y=model(X,W)

#Se define la constante de aprendizaje
mu=0.01

#El coste es la media de la raiz cuadrada de la salida real menos la desada
cost = T.mean(T.sqrt(y-Y))
#Para calcular el gradieente se le pasan los costes y los pesos
gradient =T.grad(cost=cost,wrt=W)
#Para actualizar los pesos se crea una lista con los pesos y los pesos menos el gradiente por la cte de aprendizake
#Define como actualizar los parametros
updates=[[W, W - gradient*mu]]

#Se crea la funcion de entrenar, donde se le pasan los pesos, la salida es el coste,
train=theano.function(inputs=[X,Y],outputs=cost,updates=updates,allow_input_downcast=True)

#Durante 100 iteraciones , se clacula el entreamiento
#for i in range(100):
 #   for x,y in zip(train_x,train_y): No se tinen los zip de train_x ytrain_y
  #      train(x,y)