# En este archivo de python se va a utilizar la base de datos de FRAV y Casia de imagenes  y se va a seguir la configuracion de la CNN del paper 'Learn convolutional neural network for face anti-spoofing'
import matplotlib as mpl
mpl.use('Agg')
import numpy
import timeit
from pylab import *
from logistic_sgd import LogisticRegression
from layers_gaussian_init import *
import sys
import theano
import theano.tensor
import pickle
import os, argparse
from classifiers import *
from ISOmetrics import *
import auxiliar_functions
from softmax_concatenated import sgd_optimization
from random import shuffle
#nkerns=[96, 256, 386, 384, 256]
from shuffle_genuine_attack_data import *

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        param_update = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)

        updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
        updates.append((param, param - learning_rate * param_update))

    return updates

def evaluate_lenet5(learning_rate_init=0.009, n_epochs=200, nkerns=[96, 256, 386, 384, 256], batch_size=20):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints nkerns=[96, 256, 386, 384, 256] changed to: [80, 200, 250, 250, 150]
    :param nkerns: number of kernels on each layer
    """
    argumento = argparse.ArgumentParser()

    argumento.add_argument('-i', '--i', required=True, help="path where the pkl dabase file is")
    argumento.add_argument("-o", "--o", required=True,help="path where save the output")

    # Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['i']

    # Se le asigna el segundo argumento ala ruta de salida
    out_path = args['o']
    orig_stdout = sys.stdout
    f = file(out_path+'out.txt', 'w')#'out.txt', 'w')
    sys.stdout = f

    print ('In this file are the results of using casia architecture and FRAV image database')
    print ('In the architecture conv, pool, response normalization,fully connect, dropout and softmax layers are used with relu. No strides are used')
    print ('The configuration of the net is', 'learning_rate=', learning_rate_init, 'n_epochs=', n_epochs, 'nkerns=', nkerns, 'batch_size=', batch_size)
    print ('Early stop has been deleted')
    print('For training has been used softmax classifier and for testing softmax and SVM')
    print ('In this example, two classes are going to be used, class 0 for real users and class 1 for attacks')

    print ('Start reading the data...')
    rng = numpy.random.RandomState(123456)

	
    open_file2 = open(in_path, 'rb')
    data = pickle.load(open_file2)
    open_file2.close()

    X_train, y_train, name_train, X_test, y_test, name_test = aleatorizar_muestras_train_test(data)
    del data
    eliminar_train = len(X_train)%batch_size

    X_train = X_train[:-eliminar_train]
    y_train = y_train[:-eliminar_train]
    name_train = name_train[:-eliminar_train]

    eliminar_test = len(X_test)%batch_size

    X_test = X_test[:-eliminar_test]
    y_test = y_test[:-eliminar_test]
    name_test = name_test[:-eliminar_test]

    train = list(zip(X_train,y_train, name_train))
    shuffle(train)
    train_set_x, y_train, name_train = zip(*train)

    test = list(zip(X_test,y_test, name_test))
    shuffle(test)
    test_set_x, y_test, name_list = zip(*test)
    del X_train, X_test
    train_set_x = theano.shared(numpy.array(train_set_x, dtype= 'float32'), borrow=True)
    test_set_x = theano.shared(numpy.array(test_set_x, dtype= 'float32'), borrow=True)
    train_set_y = theano.shared(numpy.array(y_train, dtype='int32'), borrow=True)
    test_set_y = theano.shared(numpy.array(y_test, dtype='int32'), borrow=True)
    valid_set_x = test_set_x
    valid_set_y = test_set_y

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    print("n_train_samples: %d" % n_train_batches)
    print("n_valid_samples: %d" % n_valid_batches)
    print("n_test_samples: %d" % n_test_batches)
    print("n_batches:")

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    print("n_train_batches: %d" % n_train_batches)
    print("n_valid_batches: %d" % n_valid_batches)
    print("n_test_batches: %d" % n_test_batches)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    is_train = T.iscalar('is_train')  # To differenciate between train and test
    l_r = theano.shared(np.array(learning_rate_init, dtype=theano.config.floatX))
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)http://deeplearning.net/software/theano/library/tensor/raw_random.html
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 128, 128))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape(batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 128, 128),
        filter_shape=(nkerns[0], 3, 11, 11),
        stride=(1, 1),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer1 = LeNetConvPoolLRNLayer(
        rng,
        input=layer0.output,
        # image_shape=(batch_size, nkerns[0], 27, 27),
        image_shape=(batch_size, nkerns[0], 59, 59),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        lrn=True,
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 28, 28),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu
    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 26, 26),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        activation=theano.tensor.nnet.relu
    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 24, 24),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2),
        activation=theano.tensor.nnet.relu
    )

    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer5 = Fully_Connected_Dropout(
        rng,
        input=layer5_input,
        n_in=nkerns[4] * 11 * 11,
        n_out=3000,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer6 = Fully_Connected_Dropout(
        rng,
        input=layer5.output,
        n_in=3000,
        n_out=3000,
        is_train=is_train,
        activation=theano.tensor.nnet.relu
    )

    layer7 = FullyConnected(
        rng,
        input=layer6.output,
        n_in=3000,
        n_out=1000,
        activation=theano.tensor.nnet.relu
    )

    layer8 = LogisticRegression(input=layer7.output, n_in=1000, n_out=2)

    salidas_capa8 = theano.function(
        [index],
        layer8.y_pred,
        on_unused_input='ignore',
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    salidas_probabilidad = theano.function(
        [index],
        layer8.p_y_given_x,
        on_unused_input='ignore',

        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    salidas_capa7_test = theano.function(
        [index],
        layer7.output,
        on_unused_input='ignore',
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    salidas_capa7_valid = theano.function(
        [index],
        layer7.output,
        on_unused_input='ignore',
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )
    salidas_capa7_train = theano.function(
        [index],
        layer7.output,
        on_unused_input='ignore',
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](1)
        }
    )


    # the cost we minimize during training is the NLL of the model
    cost = layer8.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )

    a = T.iscalar('is_train')

    update2 = []
    update2.append((l_r,l_r*0.995))

    update_l_rate = theano.function([a], a+1, updates = update2)

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)

        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params + layer7.params + layer8.params
    momentum = 0.9
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    ## Learning rate update

    updates = [
        (param_i, param_i - l_r.get_value() * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        #updates=gradient_updates_momentum(cost, params, l_r.get_value(), momentum),
        givens={
            x: train_set_x[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            y: train_set_y[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            is_train: np.cast['int32'](1)
        }
    )
    # and in the training loop
    #cost_ij = train_model(minibatch_index, learning_rate)


    ###############
    # TRAIN MODEL #
    ###############
    print ('... training')
    print (' ')
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2) # In each epoch

    print("patience: %d" % patience)
    print("patience_increase: %d" % patience_increase)
    print("improvement threshold: %d" % improvement_threshold)
    print("validation_frequency: %d" % validation_frequency)
    print (' ')

    # go through this many minibatche before checking the network
    # on the validation set; in this case we check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = timeit.default_timer()
    error_epoch = []
    lista_coste = []
    epoch = 0
    done_looping = False
    learning_Rate_list = []
    momentum = 0.9

    print ('n_train_batches', n_train_batches)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)

            cost_ij = train_model(minibatch_index)
            lista_coste.append(cost_ij)
            learning_Rate_list.append(l_r.get_value())

            if (iter + 1) % validation_frequency == 0:


                # compute zero-one loss on validation set

                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%, cost %f, l. rate %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., cost_ij,l_r.get_value()))

                error_epoch.append(this_validation_loss * 100)
                updating_l_rate = update_l_rate(1)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    w0_test = layer0.W.get_value()
                    b0_test = layer0.b.get_value()

                    w1_test = layer1.W.get_value()
                    b1_test = layer1.b.get_value()

                    w2_test = layer2.W.get_value()
                    b2_test = layer2.b.get_value()

                    w3_test = layer3.W.get_value()
                    b3_test = layer3.b.get_value()

                    w4_test = layer4.W.get_value()
                    b4_test = layer4.b.get_value()

                    w5_test = layer5.W.get_value()
                    b5_test = layer5.b.get_value()

                    w6_test = layer6.W.get_value()
                    b6_test = layer6.b.get_value()

                    w7_test = layer7.W.get_value()
                    b7_test = layer7.b.get_value()

                    w8_test = layer8.W.get_value()  # Los guardo para hacer el solftmax como clasificador
                    b8_test = layer8.b.get_value()

                    input_TrainClass = []

                    # sal_capa2 = [salidas_capa2_train(i) for i in xrange(n_train_batches)]
                    sal_capa2 = [salidas_capa7_train(i) for i in range(n_train_batches)]
                    for i in sal_capa2:
                        for j in i:
                            input_TrainClass.append(j)




    ###############################
    ###    TESTING MODEL        ###
    ###############################
    # Aqui se tiene que cargar la red

    layer0.W.set_value(w0_test)
    layer0.b.set_value(b0_test)

    layer1.W.set_value(w1_test)
    layer1.b.set_value(b1_test)

    layer2.W.set_value(w2_test)
    layer2.b.set_value(b2_test)

    layer3.W.set_value(w3_test)
    layer3.b.set_value(b3_test)

    layer4.W.set_value(w4_test)
    layer4.b.set_value(b4_test)

    layer5.W.set_value(w5_test)
    layer5.b.set_value(b5_test)

    layer6.W.set_value(w6_test)
    layer6.b.set_value(b6_test)

    layer7.W.set_value(w7_test)
    layer7.b.set_value(b7_test)

    layer8.W.set_value(w8_test)  # Se cargan para el softmax como clasificador
    layer8.b.set_value(b8_test)

    input_test = []
    input_valid = []
    sal_capa7 = [salidas_capa7_test(i) for i in range(n_test_batches)]

    for i in sal_capa7:
        for j in i:
            input_test.append(j)


    y_train = y_train[0:len(input_TrainClass)]
    y_test = y_test[0:len(input_test)]



 ########## SVM ###########
    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(input_TrainClass, y_train, input_test, y_test)
    print ('SVM RBF scores:', scores_SVM)

    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_RBF-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'SVM_RBF-')
    #
    # ########## SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(input_TrainClass, y_train, input_test, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_LINEAR-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'SVM_linear-')

    #
    ############  KNN  ##############
    knn_pred, knn_pred_prob, scores_knn = KNNClas(input_TrainClass, y_train, input_test, y_test)
    #
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path+'KNN-', name_list)
    # DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'KNN-')
    #
    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(input_TrainClass, y_train, input_test, y_test)
    print('Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'DecisionTree-', name_list)
    # DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'TREE-')
    #
    #
    # ########### SOFTMAX ############
    print('----------SOFTMAX---------')

    y_pred_junto = []
    y_prob_junto = []

    # test it on the test set
    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = numpy.mean(test_losses)

    for i in range(n_test_batches):
        y_pred_test = salidas_capa8(i)
        y_probabilidad = salidas_probabilidad(i)

        for j in y_pred_test:
            y_pred_junto.append(j)

        for j in y_probabilidad:
            y_prob_junto.append(j[0])

    print((' test error of best model %f %%') % (test_score * 100.))

    print ('SOFTMAX scores:', test_score)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, y_pred_junto, y_prob_junto, out_path+'SOFTMAX-', name_list)
    # DETCurve(y_test, y_prob_junto, 0, out_path+'SOFTMAX-')
    metric(y_prob_junto, y_pred_junto, y_test, 0, out_path+'SOFTMAX-')


    ############## PCA ################

    X_train_after_PCA, X_test_after_PCA, X_valid_after_PCA = PCAClas(input_TrainClass, y_train, input_test, y_test, input_test)

    ########## PCA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_RBF-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0,out_path+'PCA_SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'PCA_SVM_RBF-')

    ########## PCA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(X_train_after_PCA, y_train, X_test_after_PCA, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_LINEAR-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'PCA_SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'PCA_SVM_Linear-')

    ############  PCA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'PCA_KNN-', name_list)
    # DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'PCA_KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'PCA_KNN-')

    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('PCA Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'PCA_DecisionTree-', name_list)
    # DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'PCA_TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'PCA_TREE-')

    ########### SOFTMAX + PCA ############
    print('----------SOFTMAX + PCA---------')
    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [X_train_after_PCA, X_valid_after_PCA, X_test_after_PCA , y_train, y_test, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data, len(X_train_after_PCA[0]), 0)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
	# Como es el error, buscamos el argmin , no el argmax como en los otros casos
    indice_softmax = np.argmin(test_scores)
    print('Vamos a testear el sotmax:', test_scores, indice_softmax)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'PCA_Softmax-', name_list)
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 0, out_path+'PCA_Softmax-')
    print('best softmax PCA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])
    del data
    ##############    LDA   ###################
    X_train_after_LDA, X_test_after_LDA, X_valid_after_LDA = LDAClas(input_TrainClass, y_train, input_test, y_test, input_test)

    ########## LDA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_RBF-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0,out_path+'LDA_SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'LDA_SVM_RBF-')


    ########## LDA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(X_train_after_LDA, y_train, X_test_after_LDA, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_LINEAR-', name_list)
    # DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'LDA_SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'LDA_SVM_linear-')

    ############  LDA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'LDA_KNN-', name_list)
    # DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'LDA_KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'LDA_KNN-')

    ##########  LDA DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('LDA-Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'LDA_DecisionTree-', name_list)
    # DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'LDA_TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'LDA_TREE-')

    ########### SOFTMAX + LDA ############
    print('----------SOFTMAX + LDA---------')
    learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [X_train_after_LDA, X_valid_after_LDA, X_test_after_LDA , y_train, y_test, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data, len(X_train_after_LDA[0]), 0)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
	# Como es el error, buscamos el argmin , no el argmax como en los otros casos
    indice_softmax = np.argmin(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'LDA_Softmax-', name_list)
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 0, out_path+'LDA_Softmax-')
    print('best softmax LDA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])
    del data
   ########  FIN CLASIFICADORES #########
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    plt.clf()
    plt.plot(error_epoch)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.savefig(out_path+'error.png')

    plt.clf()
    plt.plot(lista_coste)
    plt.ylabel('cost_ij')
    plt.xlabel('iteration')
    plt.savefig(out_path+'cost.png')

    plt.clf()
    plt.plot(learning_Rate_list)
    plt.ylabel('Learning rate')
    plt.xlabel('iteration')
    plt.savefig(out_path+'LearningRate.png')


    np.save(out_path+'cost', lista_coste)
    np.save(out_path+'error', error_epoch)
    np.save(out_path + 'Learning_Rate', learning_Rate_list)

    end_time = timeit.default_timer()
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
