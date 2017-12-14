import numpy as np
import matplotlib.pyplot as plt
import os, argparse

argumento = argparse.ArgumentParser()

argumento.add_argument('-i_p', '--i_p', required=True, help="path where the predicted probabilities are")
argumento.add_argument('-i_r', '--i_r', required=True, help="path where the real labels are")
argumento.add_argument("-o", "--o", required=True, help="path where save the output")

# Se le asigna el primer argumento a la ruta de entrada
args = vars(argumento.parse_args())
in_path_pred = args['i_p']
in_path_real = args['i_r']
out_path = args['o']


y_real = np.load(in_path_real)
y_pred= np.load(in_path_pred)

try:
    y_predict=y_pred[:,0]
except:
    y_predict = y_pred

clase_y_prob = 0 #La positiva, la real, la buena

# FN_vect = np.zeros([rang_max+1])  # real 0 pred 1
# FP_vect = np.zeros([rang_max+1])   # real 1 pred 0
# TP_vect = np.zeros([rang_max+1])
# TN_vect = np.zeros([rang_max+1])
#
# thres_range = np.arange(0,1+(1./rang_max), (1./rang_max))
#
# for position_gen, th in enumerate(thres_range):

thersholds = np.sort(y_predict)

FN_vect = np.zeros([len(thersholds)])  # real 0 pred 1
FP_vect = np.zeros([len(thersholds)])   # real 1 pred 0
TP_vect = np.zeros([len(thersholds)])
TN_vect = np.zeros([len(thersholds)])

for position_gen, th in enumerate(thersholds):

    FP = 0.
    TN = 0.
    TP = 0.
    FN = 0.

    for pos, value in enumerate(y_predict):
        if value >= th:
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

    FP_vect[position_gen] = FP
    TP_vect[position_gen] = TP
    FN_vect[position_gen] = FN
    TN_vect[position_gen] = TN

FAR = FP_vect/(FP_vect+TN_vect)
FRR = FN_vect/(FN_vect+TP_vect)

np.save(out_path + 'FAR_S', FAR)
np.save(out_path + 'FRR_s', FRR)

comparative = 10

for iter in range(0,len(thersholds)):
    resta = abs(FAR[iter] - FRR[iter])
    if resta < comparative:
        comparative = resta
        save_resta = resta
        FAR_EER_value = FAR[iter]
        FRR_EER_value = FRR[iter]
        iter_saved = iter

thres_saved = float(iter_saved)/float(len(thersholds))

print ('FAR_EER_value',FAR_EER_value)
print ('FRR_EER_value',FRR_EER_value)
print ('comparative',comparative)
print ('resta', save_resta)
print ('iter', iter_saved, 'threshold',thres_saved)

max=round(np.amax([FAR_EER_value,FRR_EER_value]),3)
min=round(np.amin([FAR_EER_value,FRR_EER_value]),3)

EER = (FAR_EER_value+FRR_EER_value)/2

print('THe EER value for', in_path_pred, 'is', EER, 'max y min', max, min)
#
# letras = 'EER = '+str(round((FAR_EER_value+FRR_EER_value)/2,3)) + '\n' +'threshold = ' + str(thres_saved)
#
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# plt.clf()
# plt.text(0.98,0.62, letras,
#          horizontalalignment='right',bbox=props )
# plt.plot(thres_range, FAR, lw=2, color='navy', label='FAR')
# plt.plot(thres_range, FRR, lw=2, color='red', label='FRR')
#
#
# plt.xlabel('Threshold')
# plt.title('FAR-FRR')
# plt.legend(loc="upper right")
# plt.savefig('CAsia_vid_PCA_SOFTMAX_SVM_RBF_FAR_FRR.png', bbox_inches='tight')

