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
from casia_Lenet_multi_1ch import *
from casia_Lenet_multi_3ch import *


def eliminar_data(X_train_rgb, X_train_nir, X_train_thr, X_test_rgb, X_test_nir, X_test_thr, y_train, y_test, name_train, name_test,batch_size):
    eliminar_train = len(X_train_rgb)%batch_size

    X_train_rgb = X_train_rgb[:-eliminar_train]
    y_train = y_train[:-eliminar_train]
    name_train = name_train[:-eliminar_train]
    X_train_nir = X_train_nir[:-eliminar_train]
    X_train_thr = X_train_thr[:-eliminar_train]

    eliminar_test = len(X_test_rgb)%batch_size

    X_test_rgb = X_test_rgb[:-eliminar_test]
    X_test_nir = X_test_nir[:-eliminar_test]
    X_test_thr = X_test_thr[:-eliminar_test]

    y_test = y_test[:-eliminar_test]
    name_test = name_test[:-eliminar_test]

    train = list(zip(X_train_rgb, X_train_nir, X_train_thr,y_train, name_train))
    shuffle(train)
    X_train_rgb, X_train_nir, X_train_thr, y_train, name_train = zip(*train)

    test = list(zip(X_test_rgb, X_test_nir, X_test_thr ,y_test, name_test))
    shuffle(test)
    X_test_rgb, X_test_nir, X_test_thr, y_test, name_list = zip(*test)

    return X_train_rgb, X_train_nir, X_train_thr, y_train, name_train, X_test_rgb, X_test_nir, X_test_thr, y_test, name_list

def evaluate_lenet5_clas(learning_rate_init=0.009, n_epochs=230, nkerns=[56, 156, 256, 254, 106], batch_size=30):
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
    start_time = timeit.default_timer()

    argumento = argparse.ArgumentParser()

    argumento.add_argument('-i', '--i', required=True, help="path where the pkl dabase file is")
    argumento.add_argument("-o", "--o", required=True,help="path where save the output")

    # Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['i']

    # Se le asigna el segundo argumento ala ruta de salida
    out_path = args['o']

    rng = numpy.random.RandomState(123456)

    open_file2 = open(in_path, 'rb')
    data = pickle.load(open_file2)
    open_file2.close()

    rgb_X_train, nir_X_train, thr_X_train, y_train, name_train, rgb_X_test,nir_X_test,thr_X_test, y_test, name_test = aleatorizar_muestras_train_test_multi(data)
    del data

    rgb_X_train, nir_X_train, thr_X_train, y_train, name_train, rgb_X_test, nir_X_test, thr_X_test, y_test, name_list = eliminar_data(rgb_X_train, nir_X_train, thr_X_train, rgb_X_test, nir_X_test, thr_X_test, y_train, y_test, name_train, name_test,batch_size)


    # Get train, valid and test features of layer 7
    RGB_train_x, RGB_test_x = evaluate_lenet5_multi_3ch(learning_rate_init, n_epochs, nkerns, batch_size, [rgb_X_train, rgb_X_test, y_train, y_test, name_train, name_list], out_path+'RGB_multi/')
    NIR_train_x, NIR_test_x = evaluate_lenet5_multi_1ch(learning_rate_init, n_epochs, nkerns, batch_size, [nir_X_train, nir_X_test, y_train, y_test, name_train, name_list], out_path+'NIR_multi/')
    THR_train_x, THR_test_x = evaluate_lenet5_multi_1ch(learning_rate_init, n_epochs, nkerns, batch_size, [thr_X_train, thr_X_test, y_train, y_test, name_train, name_list], out_path+'THR_multi/')

    del rgb_X_train, nir_X_train, thr_X_train, name_train, rgb_X_test, nir_X_test, thr_X_test

    # Concatenate the RGB and NIR outputs
    input_TrainClass_concatenated = np.concatenate([RGB_train_x, NIR_train_x, THR_train_x], axis=1)
    del RGB_train_x, THR_train_x, NIR_train_x
    input_TestClass_concatenated = np.concatenate([RGB_test_x, NIR_test_x, THR_test_x], axis=1)

    del  NIR_test_x, RGB_test_x, THR_test_x

	# Get y as the same length
    y_train = y_train[0:len(input_TrainClass_concatenated)]
    y_test = y_test[0:len(input_TestClass_concatenated)]

    out_path = out_path+'/frav_multi_clas/'

    orig_stdout = sys.stdout
    f = file(out_path+'out.txt', 'w')# 'out.txt', 'w')
    sys.stdout = f

    print('EL ENTRENAMIENTO DE ESTA EJECUCION SON LA DE RGB; NIR Y THR POR SEPARADO. LOS MODELOS DE RED ESCOGUDOS SON LOS MEJORES DE ESAS EJECIONES QUE SE PUEDEN VER EN SUS CARPETAS RESPETIVAS')

      ########## SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_RBF-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'SVM_RBF-')

    ########## SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path+'SVM_LINEAR-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'SVM_linear-')

    ############  KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)

    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path+'KNN-', name_list)
    DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'KNN-')

    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test)
    print('Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'DecisionTree-', name_list)
    DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'TREE-')


    ########### SOFTMAX ############

    print('----------SOFTMAX ---------')
    learning_rates = [0.005, 0.001, 0.05, 0.01]
    iterarions = 400
    data = [input_TrainClass_concatenated, input_TestClass_concatenated, input_TestClass_concatenated , y_train, y_test, y_test]
    validation_losses = np.zeros(len(learning_rates))
    test_scores = np.zeros(len(learning_rates))
    iters = np.zeros(len(learning_rates))
    Softmax_predictions = []
    Softmax_probabilities = []
    for i, lr in enumerate(learning_rates):
        validation_losses[i], test_scores[i], iters[i], predictions_aux, probabilities_aux = sgd_optimization(lr, 400, batch_size, data,len(input_TrainClass_concatenated[0]), 0)
        Softmax_predictions.append(predictions_aux)
        Softmax_probabilities.append(probabilities_aux)
    indice_softmax = np.argmin(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'SOFTMAX-', name_list)
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 0, out_path+'SOFTMAX-')
    print('best softmax scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])


    ############## PCA ################

    X_train_after_PCA, X_test_after_PCA, X_valid_after_PCA = PCAClas(input_TrainClass_concatenated, y_train, input_TestClass_concatenated, y_test, input_TestClass_concatenated)

    ########## PCA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_RBF-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0,out_path+'PCA_SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'PCA_SVM_RBF-')

    ########## PCA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(X_train_after_PCA, y_train, X_test_after_PCA, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'PCA_SVM_LINEAR-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'PCA_SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'PCA_SVM_Linear-')

    ############  PCA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'PCA_KNN-', name_list)
    DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'PCA_KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'PCA_KNN-')

    ########## DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_PCA, y_train, X_test_after_PCA, y_test)
    print('PCA Decision Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'PCA_DecisionTree-', name_list)
    DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'PCA_TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'PCA_TREE-')

    ########### SOFTMAX + PCA ############
    print('----------SOFTMAX + PCA---------')
    learning_rates = [0.005, 0.001, 0.05, 0.01]
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
    indice_softmax = np.argmin(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'PCA_Softmax-', name_list)
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 0, out_path+'PCA_Softmax-')
    print('best softmax PCA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])

    ##############    LDA   ###################
    X_train_after_LDA, X_test_after_LDA, X_valid_after_LDA = LDAClas(input_TrainClass_concatenated, y_train,input_TestClass_concatenated, y_test, input_TestClass_concatenated)

    ########## LDA SVM ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_rbf(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print ('SVM RBF scores:', scores_SVM)
    TP, TN, FP, FN =auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_RBF-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0,out_path+'LDA_SVM_RBF-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'LDA_SVM_RBF-')


    ########## LDA SVM LINEAR ###########

    SVM_pred, SVM_pred_prob, scores_SVM = SVMClas_linear(X_train_after_LDA, y_train, X_test_after_LDA, y_test)

    print ('SVM linear scores:', scores_SVM)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, SVM_pred, SVM_pred_prob, out_path + 'LDA_SVM_LINEAR-', name_list)
    DETCurve(y_test, SVM_pred_prob[:,0], 0, out_path+'LDA_SVM_linear-')
    metric(SVM_pred_prob[:,0], SVM_pred, y_test, 0, out_path+'LDA_SVM_linear-')

    ############  LDA KNN  ##############

    knn_pred, knn_pred_prob, scores_knn = KNNClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('KNN mean accuracy: ', scores_knn)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, knn_pred, knn_pred_prob, out_path + 'LDA_KNN-', name_list)
    DETCurve(y_test, knn_pred_prob[:,0], 0, out_path+'LDA_KNN-')
    metric(knn_pred_prob[:,0], knn_pred, y_test, 0, out_path+'LDA_KNN-')

    ##########  LDA DECISION TREE ###########
    tree_pred, tree_pred_prob, scores_tree = DecisionTreeClas(X_train_after_LDA, y_train, X_test_after_LDA, y_test)
    print('LDA-Tree mean accuracy: ', scores_tree)
    TP, TN, FP, FN = auxiliar_functions.analize_results(y_test, tree_pred, tree_pred_prob, out_path + 'LDA_DecisionTree-', name_list)
    DETCurve(y_test, tree_pred_prob[:,0], 0, out_path+'LDA_TREE-')
    metric(tree_pred_prob[:,0], tree_pred, y_test, 0, out_path+'LDA_TREE-')

    ########### SOFTMAX + LDA ############
    print('----------SOFTMAX + LDA---------')
    learning_rates = [0.005, 0.001, 0.05, 0.01]
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
    indice_softmax = np.argmin(test_scores)
    auxiliar_functions.analize_results(y_test, Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], out_path+'LDA_Softmax-', name_list)
    metric(Softmax_predictions[indice_softmax], Softmax_probabilities[indice_softmax], y_test, 0, out_path+'LDA_Softmax-')
    print('best softmax LDA scores with a learning rate of', learning_rates[indice_softmax], 'best validation score:', validation_losses[indice_softmax],'at iteration', iters[indice_softmax], 'and test loss:', test_scores[indice_softmax])



   ########  FIN CLASIFICADORES #########


    end_time = timeit.default_timer()
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    evaluate_lenet5_clas()








