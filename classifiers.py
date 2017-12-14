from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import logspace



def SVMClas_rbf(X_train, y_train, X_test, y_test):
    ########## SVM ###########

    print('----------SVM results ---------  KERNEL = rbf')
    svm_scores = []
    optimaG_list = []
    C_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]
    Gamma = logspace(-9, 3, 10)

    for Ci in C_range:
        for G in Gamma:
            SVMcla = SVC(C=Ci, gamma = G, kernel = 'rbf')
            scores = cross_val_score(SVMcla, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
            svm_scores.append(scores.mean())

    max_score = svm_scores.index(max(svm_scores))
 
    optimaC = C_range[max_score/10]
    optimaG = Gamma[max_score%10]

    print('optimaC', optimaC, 'optimaG', optimaG)

    SVM2 = SVC(C=optimaC, gamma = optimaG, kernel = 'rbf', probability=True)
    SVM2.fit(X=X_train, y=y_train)

    SVM_pred = SVM2.predict(X_test)
    SVM_pred_prob = SVM2.predict_proba(X_test)
    scores_SVM = SVM2.score(X_test, y_test)

    return SVM_pred, SVM_pred_prob, scores_SVM

def SVMClas_linear(X_train, y_train, X_test, y_test):
    ########## SVM ###########

    print('----------SVM results ---------  KERNEL = linear')
    svm_scores = []
    C_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10]

    for Ci in C_range:
        SVMcla = SVC(C=Ci, kernel = 'linear')
        scores = cross_val_score(SVMcla, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        svm_scores.append(scores.mean())

    optimaC = C_range[svm_scores.index(max(svm_scores))]
    print('optimaC', optimaC)

    SVM2 = SVC(C=optimaC, kernel='linear', probability=True)
    SVM2.fit(X=X_train, y=y_train)

    SVM_pred = SVM2.predict(X_test)
    SVM_pred_prob = SVM2.predict_proba(X_test)
    scores_SVM = SVM2.score(X_test, y_test)

    return SVM_pred, SVM_pred_prob, scores_SVM

def KNNClas(X_train, y_train, X_test, y_test):

    ############  KNN  ##############

    print('---------- KNN results ---------')
    k_scores = []
    k_range = range(1, 33, 2)
    k_malas =[] 
    for i in k_range:
        try:
            neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
            scores = cross_val_score(neigh, X_train, y_train, cv=5,
                                     scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
            k_scores.append(scores.mean())
        except:
            k_malas.append(i)

    print('k con las que no se ha podido calcular:', k_malas)
    optim_position = k_scores.index(max(k_scores))  # esto te da el indice
    optimk = k_range[optim_position]
    print('optim k value:', optimk)

    neigh = neighbors.KNeighborsClassifier(n_neighbors=(optimk))
    neigh.fit(X_train, y_train)

    knn_pred_prob = neigh.predict_proba(X_test)
    knn_pred = neigh.predict(X_test)
    scores_knn = neigh.score(X_test, y_test)

    return knn_pred, knn_pred_prob, scores_knn


def PCAClas(X_train, y_train, X_test, y_test, X_valid):

    ############## PCA ################

    print('---------- PCA results ---------')

    ncomponentes = range(3, 500, 10)
    PCA_scores = []
    pca = PCA()
    com_malas = []

    for n in ncomponentes:
        try:
            pca.n_components = n
            scores = cross_val_score(pca, X_train, y_train, cv=10)  # Con neg_log_loss el predict tiene que ser con probabilidad
            PCA_scores.append(scores.mean())

        except:
            com_malas.append(n)

    print('PCA componentes con las que no se ha podido calcular:', com_malas)
    optimaNcomponentes = ncomponentes[PCA_scores.index(max(PCA_scores))]
    print('PCA optimous componentes number', optimaNcomponentes)

    pca = PCA(n_components=optimaNcomponentes)
    X_train_after_PCA = pca.fit_transform(X_train)
    X_test_after_PCA = pca.transform(X_test)
    X_valid_after_PCA = pca.transform(X_valid)

    return X_train_after_PCA, X_test_after_PCA, X_valid_after_PCA


def LDAClas(X_train, y_train, X_test, y_test, X_valid):

    ############## LDA ################

    print('---------- LDA results ---------')

    ncomponentes = range(1, 50, 5)
    LDA_scores = []
    lda = LDA()
    com_malas = []

    for n in ncomponentes:
        print ('n', n)
        try:
            lda.n_components = n
            scores = cross_val_score(lda, X_train, y_train, cv=10)  # Con neg_log_loss el predict tiene que ser con probabilidad
            LDA_scores.append(scores.mean())

        except:
            com_malas.append(n)

    print('PCA componentes con las que no se ha podido calcular:', com_malas)
    optimaNcomponentes = ncomponentes[LDA_scores.index(max(LDA_scores))]
    print('LDA optimous componentes number', optimaNcomponentes)

    lda = LDA(n_components=optimaNcomponentes)
    X_train_after_LDA = lda.fit_transform(X_train, y_train)
    X_test_after_LDA = lda.transform(X_test)
    X_valid_after_LDA = lda.transform(X_valid)

    print('max lda scores:', max(LDA_scores)) 
    return X_train_after_LDA, X_test_after_LDA, X_valid_after_LDA


def DecisionTreeClas(X_train, y_train, X_test, y_test):
    ############ DECISION TREE ##############

    print('---------- DECISION TREES results ---------')

    depth_scores = []
    depth_range = range(2, 32, 2)
    for i in depth_range:
        dTree = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(dTree, X_train, y_train, cv=5,
                                 scoring='accuracy')  # Con neg_log_loss el predict tiene que ser con probabilidad
        depth_scores.append(scores.mean())

    optim_position = depth_scores.index(max(depth_scores))  # esto te da el indice
    optim_depth = depth_range[optim_position]

    print('optim depth:', optim_depth)

    dTree = DecisionTreeClassifier(max_depth=optim_depth)
    dTree.fit(X_train, y_train)

    tree_prob = dTree.predict_proba(X_test)
    tree_pred = dTree.predict(X_test)
    scores_tree = dTree.score(X_test, y_test)

    return tree_pred, tree_prob, scores_tree

