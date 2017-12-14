from numpy import zeros, ones
from numpy import append as npappend
from random import shuffle

def aleatorizar_muestras_train_test(data):
    # En esta funcion se carga un vector de tipo[genuin_users, los nombres de los geuine users, attacks, nombres_attacks]
    # Se va a devolver muestras de train y muestras de test. En el train habra un 70% seguro de las musetras genuinas y attacks y en el test un 30% de las genuinas y un 30% de las de attacks

    # unpack data
    [genuine_list, genuine_name_list, attack_list, attack_name_list] = data

    lenght_genuine = len(genuine_list)
    lenght_attack = len(attack_list)

    # shuffle lists
    genuine = list(zip(genuine_list, genuine_name_list))
    attack = list(zip(attack_list, attack_name_list))

    shuffle(genuine)
    shuffle(attack)

    genuine_list, genuine_name_list = zip(*genuine)
    attack_list, attack_name_list = zip(*attack)

    # Asociamos el 30% de cada lista (genuine y attack) al test
    # Asociamos la parte de genuine
    X_test = genuine_list[0:int(round(0.3*len(genuine_list)))]
    name_test = genuine_name_list[0:int(round(0.3*len(genuine_name_list)))]
    y_test = zeros(len(X_test))

    # Asociamos la parte de  attacks
    auxxx = attack_list[0:int(round(0.3*len(attack_list)))]
    auxxx_name = attack_name_list[0:int(round(0.3*len(attack_name_list)))]

    X_test = X_test + auxxx
    name_test = name_test + auxxx_name
    aux_test = ones(len(auxxx))

    y_test = list(npappend(y_test, aux_test))

    assert(len(y_test)==len(X_test))

    # Asociamos el 30% de cada lista (genuine y attack) al train
    # Asociamos la parte de genuine
    X_train = genuine_list[int(round(0.3*len(genuine_list))):len(genuine_list)]
    name_train = genuine_name_list[int(round(0.3*len(genuine_name_list))):len(genuine_name_list)]
    y_train = zeros(len(X_train))

    # Asociamos la parte de  attacks
    auxxx2 = attack_list[int(round(0.3*len(attack_list))):len(attack_list)]
    auxxx2_name = attack_name_list[int(round(0.3*len(attack_name_list))):len(attack_name_list)]

    X_train = X_train + auxxx2
    name_train = name_train + auxxx2_name
    aux_train = ones(len(auxxx2))

    y_train = list(npappend(y_train, aux_train))

    assert(len(y_train)==len(X_train))

    assert (len(y_train)+len(y_test) == len(genuine_list)+len(attack_list))

    print('longitudes:', len(y_train), len(X_train[0]))
    return(X_train, y_train, name_train, X_test, y_test, name_test)



def aleatorizar_muestras_train_test_multi(data):
    # unpack data
    rgb_data, nir_data, thr_data = data
    del data

    [rgb_genuine_list, rgb_genuine_name_list, rgb_attack_list, rgb_attack_name_list] = rgb_data
    [nir_genuine_list, nir_genuine_name_list, nir_attack_list, nir_attack_name_list] = nir_data
    [thr_genuine_list, thr_genuine_name_list, thr_attack_list, thr_attack_name_list] = thr_data

    del rgb_data, nir_data, thr_data

    lenght_genuine = len(rgb_genuine_list)
    lenght_attack = len(rgb_attack_list)

    # shuffle lists
    genuine = list(zip(rgb_genuine_list,nir_genuine_list,thr_genuine_list, thr_genuine_name_list))
    attack = list(zip(rgb_attack_list,nir_attack_list,thr_attack_list, thr_attack_name_list))


    shuffle(genuine)
    shuffle(attack)

    rgb_genuine_list,nir_genuine_list,thr_genuine_list, thr_genuine_name_list = zip(*genuine)
    rgb_attack_list,nir_attack_list,thr_attack_list, thr_attack_name_list = zip(*attack)

    # Asociamos el 30% de cada lista (genuine y attack) al test
    # Asociamos la parte de genuine
    rgb_X_test = rgb_genuine_list[0:int(round(0.3*len(rgb_genuine_list)))]
    name_test = rgb_genuine_name_list[0:int(round(0.3*len(rgb_genuine_name_list)))]
    y_test = zeros(len(rgb_X_test))

    nir_X_test = nir_genuine_list[0:int(round(0.3*len(nir_genuine_list)))]

    thr_X_test = thr_genuine_list[0:int(round(0.3*len(thr_genuine_list)))]


    # Asociamos la parte de  attacks
    rgb_auxxx = rgb_attack_list[0:int(round(0.3*len(rgb_attack_list)))]
    auxxx_name = rgb_attack_name_list[0:int(round(0.3*len(rgb_attack_name_list)))]

    nir_auxxx = nir_attack_list[0:int(round(0.3*len(nir_attack_list)))]

    thr_auxxx = thr_attack_list[0:int(round(0.3*len(thr_attack_list)))]


    rgb_X_test = rgb_X_test + rgb_auxxx
    name_test = name_test + auxxx_name
    aux_test = ones(len(rgb_auxxx))


    nir_X_test = nir_X_test + nir_auxxx

    thr_X_test = thr_X_test + thr_auxxx

    y_test = list(npappend(y_test, aux_test))

    assert(len(y_test)==len(rgb_X_test))

    # Asociamos el 30% de cada lista (genuine y attack) al train
    # Asociamos la parte de genuine
    rgb_X_train = rgb_genuine_list[int(round(0.3*len(rgb_genuine_list))):len(rgb_genuine_list)]
    name_train = rgb_genuine_name_list[int(round(0.3*len(rgb_genuine_name_list))):len(rgb_genuine_name_list)]
    y_train = zeros(len(rgb_X_train))

    nir_X_train = nir_genuine_list[int(round(0.3*len(nir_genuine_list))):len(nir_genuine_list)]

    thr_X_train = thr_genuine_list[int(round(0.3*len(thr_genuine_list))):len(thr_genuine_list)]


    rgb_X_train = rgb_genuine_list[int(round(0.3*len(rgb_genuine_list))):len(rgb_genuine_list)]
    name_train = rgb_genuine_name_list[int(round(0.3*len(rgb_genuine_name_list))):len(rgb_genuine_name_list)]

    nir_X_train = nir_genuine_list[int(round(0.3*len(nir_genuine_list))):len(nir_genuine_list)]

    thr_X_train = thr_genuine_list[int(round(0.3*len(thr_genuine_list))):len(thr_genuine_list)]

    # Asociamos la parte de  attacks
    rgb_auxxx2 = rgb_attack_list[int(round(0.3*len(rgb_attack_list))):len(rgb_attack_list)]
    auxxx2_name = rgb_attack_name_list[int(round(0.3*len(rgb_attack_name_list))):len(rgb_attack_name_list)]

    nir_auxxx2 = nir_attack_list[int(round(0.3*len(nir_attack_list))):len(nir_attack_list)]

    thr_auxxx2 = thr_attack_list[int(round(0.3*len(thr_attack_list))):len(thr_attack_list)]



    rgb_X_train = rgb_X_train + rgb_auxxx2
    nir_X_train = nir_X_train + nir_auxxx2
    thr_X_train = thr_X_train + thr_auxxx2

    name_train = name_train + auxxx2_name
    aux_train = ones(len(rgb_auxxx2))

    y_train = list(npappend(y_train, aux_train))

    assert(len(y_train)==len(rgb_X_train)==len(nir_X_train)==len(thr_X_train))

    assert (len(y_train)+len(y_test) == len(rgb_genuine_list)+len(rgb_attack_list))

    return(rgb_X_train, nir_X_train, thr_X_train, y_train, name_train, rgb_X_test,nir_X_test,thr_X_test, y_test, name_test)
