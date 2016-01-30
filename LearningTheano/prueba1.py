import theano
from theano import tensor as T

#Se definen las varaiables
a= T.scalar()
b= T.scalar()

#Se define la funcion simbolica, la operacion
y=a*b

#Se crea una funcion:
#nombredelafuncion=theano.function(inputs[a,b,c..], outputs=[y,..])
#en ese ejemplo:

multiply=theano.function(inputs=[a,b],outputs=y)

#Se le asignan valores reales a la funcion y se realiza la operacion
print multiply(3,2)
res=multiply(4,5)
print res