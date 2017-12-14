BBDD='multispectral_2_3/frav_multi_feat'

python FAR_FRR.py -i_p=${BBDD}'/SVM_RBF-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/SVM_RBF_'
python FAR_FRR.py -i_p=${BBDD}'/SVM_LINEAR-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/SVM_LINEAR_'
python FAR_FRR.py -i_p=${BBDD}'/KNN-y_probabilidad.npy' -i_r=${BBDD}'/KNN-y_real.npy' -o=${BBDD}'/KNN_'
python FAR_FRR.py -i_p=${BBDD}'/DecisionTree-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/DecisionTree_'
python FAR_FRR.py -i_p=${BBDD}'/SOFTMAX-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/Softmax_'

python FAR_FRR.py -i_p=${BBDD}'/PCA_SVM_RBF-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/PCA_SVM_RBF_'
python FAR_FRR.py -i_p=${BBDD}'/PCA_SVM_LINEAR-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/PCA_SVM_LINEAR_'
python FAR_FRR.py -i_p=${BBDD}'/PCA_KNN-y_probabilidad.npy' -i_r=${BBDD}'/KNN-y_real.npy' -o=${BBDD}'/PCA_KNN_'
python FAR_FRR.py -i_p=${BBDD}'/PCA_DecisionTree-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/PCA_DecisionTree_'
python FAR_FRR.py -i_p=${BBDD}'/PCA_Softmax-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/PCA_Softmax_'

python FAR_FRR.py -i_p=${BBDD}'/LDA_SVM_RBF-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/LDA_SVM_RBF_'
python FAR_FRR.py -i_p=${BBDD}'/LDA_SVM_LINEAR-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/LDA_SVM_LINEAR_'
python FAR_FRR.py -i_p=${BBDD}'/LDA_KNN-y_probabilidad.npy' -i_r=${BBDD}'/KNN-y_real.npy' -o=${BBDD}'/LDA_KNN_'
python FAR_FRR.py -i_p=${BBDD}'/LDA_DecisionTree-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/LDA_DecisionTree_'
python FAR_FRR.py -i_p=${BBDD}'/LDA_Softmax-y_probabilidad.npy' -i_r=${BBDD}'/SVM_RBF-y_real.npy' -o=${BBDD}'/LDA_Softmax_'
