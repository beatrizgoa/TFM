import theano
import theano.tensor as T

print '----------------------'
print 'Funcion'
print '---------------------'
###################
#FUNCIoN LOGISTICA:
###################

#Se define la variable simbolica en forma de matriz
x = T.dmatrix('x')

#Se define la sigmoide
s = 1 / (1 + T.exp(-x))

#Se crea la funciOn donde la entrada es x y la salida es la
#salida de la funcion
logistic = theano.function([x], s)

#Se le pasa una matriz a la funcion simbolica
RES=logistic([[0, 1], [-1, -2]])
print 'Se muestra el resultado de la funcion logistica'
print RES

print '----------------------'
print 'Varias salidas'
print '---------------------'
############################
#FUNCIONES CON VARIAS SALIDAS
############################

#Se definen variables
M=T.dmatrix('M')
N=T.dmatrix('N')

#Tambien se pueden definir asi:
M,N=T.dmatrices('M','N')


dif1=M-N
dif2=abs(M-N)
dif3=(M-N)**2 #El cuadrado de un numero se hace con ** y no con ^

#Se crea la funcion
diferencias=theano.function(inputs=[M,N], outputs=[dif1,dif2,dif3])

#Se le pasan los argumentos a la funcion
res2=diferencias([[1, 1], [1, 1]], [[0, 1], [2, 3]])

print 'Mostramos dif1'
print res2[0]

print 'Mostramos dif2'
print res2[1]

print 'Mostramos dif3'
print res2[2]

print '----------------------'
print 'Argumentos por defecto'
print '---------------------'
###############################
#PONER POR DEFECTO ARGUMENTOS
################################
#Para definir las variables tambien se pueden definir juntas
x, y,w = T.dscalars('x', 'y','w')
#Operacion:
z = (x + y) * w
#funcion
f = theano.function([x, theano.Param(y, default=1), theano.Param(w, default=2, name='w_by_name')], z)
#La y se le asigna por defecto 1, se le asigna a la entrada.

print 'entrada unica con x=10, Por defecto y=1, w=2'
r1=f(10)
print r1

print'entrada con x=10 y=5, Por defecto w=2'
r2=f(10,5)
print r2

print 'entrada con x=10 y=5 y w=1'
r3=f(10,5,w_by_name=1)
print r3

#Si se crea otra funcion y quieers llamarle a w de otra manera:
prueba=x+y+w
f2 = theano.function([x, theano.Param(y, default=3), theano.Param(w, default=2, name='Valor_w')], prueba)

print 'entrada con x=10 y=5 y w=1 en prueba'
hola=f2(10,5,Valor_w=1)
print hola


print '----------------------'
print 'variables compartidas'
print '---------------------'
#########################################
#VARIABLES COMPARTIDAS (SHARED VARIABLES)
#########################################

#Se crea la variable compartida inicializandola a cero
state = theano.shared(0)
inc = T.iscalar('inc')
acumulador = theano.function([inc], state, updates=[(state, state+inc)])
decrecesor= theano.function([inc], state, updates=[(state, state-inc)])
#Las funciones coje inc, devuelve su salida es state, y state cambia su valor de state a state + inc

a=state.get_value()
print a

aa=acumulador(2)
print aa

a=state.get_value()
print a

aa=acumulador(3)
print aa

bb=decrecesor(2)
print bb

a=state.get_value()
print a