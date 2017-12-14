import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def DETCurve(y_real, y_prob, clase_y_prob, path_out):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.

    https://jeremykarnowski.wordpress.com/2015/08/07/detection-error-tradeoff-det-curves/
    """
    FPR = np.zeros(10)
    FNR = np.zeros(10)

    for threshold in range(0,10):
        th = threshold*0.1
        FN = 0.  # real 0 pred 1
        FP = 0.  # real 1 pred 0
        TP = 0.
        TN = 0.

        for pos, value in enumerate(y_prob):

            if value  > th:
                y_pred = clase_y_prob
            else:
                y_pred = 1 - clase_y_prob

            if y_real[pos] == 0 and y_pred == 0:
                TP += 1
            if y_real[pos] == 0 and y_pred == 1:
                FN += 1
            if y_real[pos] == 1 and y_pred == 1:
                TN += 1
            if y_real[pos] == 1 and y_pred == 0:
                FP += 1

        FPR[threshold] = FP/(FP+TN)
        FNR[threshold] = FN/(TP+FN)

    print ('FPR, FNR:', FPR, FNR)
    plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.title('DET curve')
    plt.plot(FPR,FNR)
    plt.savefig(path_out+'DET_curve.png')
    plt.close()

def calculate_sumatorios(clasificacion_real, clasificacion_predict, clas_user, clas_attack):
    sum_APCER = 0
    sum_BPCER = 0

    for pos in range(0, len(clasificacion_real)):
        if clasificacion_real[pos] == clas_attack and clasificacion_predict[pos] == clas_user:  # Para APCER tine que ser los ataques
            sum_APCER += 1

        if clasificacion_real[pos] == clas_user and clasificacion_predict[pos] == clas_attack:  # Para BPCER tine que ser los usuarios (bona fides)
            sum_BPCER += 1
    return sum_APCER, sum_BPCER


def metric(probabilidades, clasificacion_predict, clasificacion_real, probabilidades_son_de_clase, out_path):

    clas_user = 0
    clas_attack = 1

    sum_APCER, sum_BPCER = calculate_sumatorios(clasificacion_real, clasificacion_predict, clas_user, clas_attack)
    # N_PAIS es como los verdaderos negativos en las etiquetas reales
    # N_BF es como los verdaderos positicos en las etiqueta reales
    a = np.where(np.array(clasificacion_real)==clas_attack)
    N_PAIS = float(len(a[0]))
    print('N_PAIS', N_PAIS)

    b = np.where(np.array(clasificacion_real)==clas_user)
    N_BF = float(len(b[0]))
    # Se calculan
    APCER = float(float(1/N_PAIS)*sum_APCER)
    BPCER = float(sum_BPCER/N_BF)
    print('N_BF:', N_BF)

    print 'APCER = ',APCER, 'BPCER = ', BPCER

    APCER_ROC = np.zeros(10)
    BPCER_ROC = np.zeros(10) # Para 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    for pos in range(0, 10):
        probabilidades_umbral = []
        umbral = pos*0.1
        for prob_i in probabilidades:
            if prob_i>umbral:
                probabilidades_umbral.append(probabilidades_son_de_clase)
            else:
                probabilidades_umbral.append(1-probabilidades_son_de_clase)

        aux_apcer, aux_bpcer = calculate_sumatorios(clasificacion_real, probabilidades_umbral, clas_user, clas_attack)
        APCER_ROC[pos] = float(float(aux_apcer)/N_PAIS)
        BPCER_ROC[pos] = float(float(aux_bpcer)/N_BF)
    np.save(out_path+'APCER', APCER_ROC)
    np.save(out_path+'BPCER', BPCER_ROC)


    plt.figure()
    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.title('APCER - BPCER curve')
    plt.plot(APCER_ROC,BPCER_ROC)
    plt.savefig(out_path+'APCER-BPCER-curve.png')
    plt.close()
