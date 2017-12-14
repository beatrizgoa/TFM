import os
import cv2
import numpy as np
from auxiliar_functions import *
from randomize_data import *

def read_videos():
    #PAths are defined
    path_train = os.path.join('cbsr_antispoofing', 'train_release')
    path_test = os.path.join('cbsr_antispoofing', 'test_release')

    #VAriables where the data is going to be saved
    X_train = []
    y_train = []

    X_test = []
    y_test = []


    print ('SE VA A LEER LOS VIDEOS DE TRAIN')
    #Folder users are looked for in train folder
    for pos, user in enumerate(os.listdir(path_train)):
        image_folder_rgb = os.path.join(path_train, user)
        print ('se van a leer los de usuario %s' % user)

        #Videos are looked for in users folder
        for pos2, video in enumerate(os.listdir(image_folder_rgb)):
            print ('se va a leer el video %s' % video)

            if pos2 == 0 or pos2 == 1: #Clase real , clase 0
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 0' % (video, pos2))
                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0
                while (captura.isOpened()):
                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 3:
                            aux_vecct_X, aux_vect_y = procesing(frame, 0) #Class 0

                            for x in aux_vecct_X:
                                X_train.append(x)

                            for y in aux_vect_y:
                                y_train.append(y)


                    if ret == False:
                            captura.release()
                captura.release()

                print('se ha acabdao de leer el video %s' % video)




            if pos2 == 2 or pos2 == 3:  # Clase mask , clase 1
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 1' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 3:
                            aux_vecct_X, aux_vect_y = procesing(frame, 1) #Cl1ss 1
                            for x in aux_vecct_X:
                                X_train.append(x)

                            for y in aux_vect_y:
                                y_train.append(y)
                    if ret == False:
                            captura.release()

                captura.release()

                print('se ha acabdao de leer el video %s' % video)





            if pos2 == 4 or pos2 == 5:  # Clase mask eyes, clase 2
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 2' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 3:

                            aux_vecct_X, aux_vect_y = procesing(frame, 2) #Class 2
                            for x in aux_vecct_X:
                                X_train.append(x)

                            for y in aux_vect_y:
                                y_train.append(y)

                    if ret == False:
                        captura.release()
                captura.release()
                print('se ha acabdao de leer el video %s ' % video)




            if pos2 == 6 or pos2 == 7:  # Clase phone , clase 3
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 3' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 3:

                            aux_vecct_X, aux_vect_y = procesing(frame, 3) #Class 3

                            for x in aux_vecct_X:
                                X_train.append(x)

                            for y in aux_vect_y:
                                y_train.append(y)

                    if ret == False:
                        captura.release()
                print('se ha acabdao de leer el video %s' % video)


    print ('SE HA ACABADO CON LOS VIDEOS DE TRAIN Y SE VA A COMENZAR CON LOS DE TEST')


    # Folder users are looked for in test folder
    for poss, user in enumerate(os.listdir(path_test)):
        image_folder_rgb = os.path.join(path_test, user)
        print ('se van a leer los de usuario %s' % user)


        # Videos are looked for in users folder
        for pos2, video in enumerate(os.listdir(image_folder_rgb)):
            print (pos2, video_file)
            print ('se va a leer el video %s' % video)

            if pos2 == 0 or pos2 == 1:  # Clase real , clase 0
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 0' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 1.5:
                            aux_vecct_X, aux_vect_y = procesing(frame, 0) #Class 0
                            for x in aux_vecct_X:
                                X_test.append(x)

                            for y in aux_vect_y:
                                y_test.append(y)

                    if ret == False:
                        captura.release()
                captura.release()
                print('se ha acabdao de leer el video %s' % video)


            if pos2 == 2 or pos2 == 3:  # Clase mask , clase 1
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 1' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 1.5:
                            aux_vecct_X, aux_vect_y = procesing(frame, 1)  # Class 1
                            for x in aux_vecct_X:
                                X_test.append(x)

                            for y in aux_vect_y:
                                y_test.append(y)
                    if ret == False:
                        captura.release()
                captura.release()

                print('se ha acabdao de leer el video %s' % video)



            if pos2 == 4 or pos2 == 5:  # Clase mask eyes, clase 2
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 2' %(video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 1.5:
                            aux_vecct_X, aux_vect_y = procesing(frame, 2)  # Class 2
                            for x in aux_vecct_X:
                                X_test.append(x)

                            for y in aux_vect_y:
                                y_test.append(y)

                    if ret == False:
                        captura.release()
                captura.release()
                print('se ha acabdao de leer el video %s' % video)

            if pos2 == 6 or pos2 == 7:  # Clase phone , clase 3
                print ('se va a leer el video %s que esta en la pos %d y es de la clase 3' % (video, pos2))

                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                count = 0

                while (captura.isOpened()):

                    #For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                        if count < 1.5:
                            aux_vecct_X, aux_vect_y = procesing(frame, 3)  # Class 3
                            for x in aux_vecct_X:
                                X_test.append(x)

                            for y in aux_vect_y:
                                y_test.append(y)

                    if ret == False:
                        captura.release()
                captura.release()
                print('se ha acabdao de leer el video %s' % video)

    print ('se ha acabado de leer los videos de test')


    data = [X_train, X_test, y_train, y_test]

    # save_file2 = open('/home/beaa/Escritorio/Theano/results/casia_videos/data_read_no_randomize.pkl', 'wb')
    # pickle.dump(data, save_file2, -1)
    # save_file2.close()

    train_index, validate_index, X_train, y_train, X_validate, y_validate = randomize(X_train, y_train, X_test, y_test, 0.3)

    data_all = [train_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate]

    # save_file2 = open('C:\Users\FRAV\Desktop\Beatriz\casiaVideo_results\data_all.pkl', 'wb')
    # pickle.dump(data_all, save_file2, -1)
    # save_file2.close()

    return train_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate


def read_videos2(path_in, resize_ratio, path_out):
    # PAths are defined
    path_train = os.path.join(path_in, 'train_release')
    path_test = os.path.join(path_in, 'test_release')

    # VAriables where the data is going to be saved
    Xlist_train = []
    ylist_train = []

    Xlist_test = []
    ylist_test = []

    name_list = []

    print('SE VA A LEER LOS VIDEOS DE TRAIN')
    # Folder users are looked for in train folder
    for pos, user in enumerate(os.listdir(path_train)):
        image_folder_rgb = os.path.join(path_train, user)
        print('se van a leer los de usuario %s' % user)
        count = -1

        # Videos are looked for in users folder
        for pos2, video in enumerate(os.listdir(image_folder_rgb)):  # pos da la clase

            if video == 'HR_1.avi' or video == 'HR_2.avi' or video == 'HR_3.avi' or video == 'HR_4.avi':
                continue
            else:

                print('se va a leer el video %s' % video)
                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                while (captura.isOpened()):
                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1

                    else:
                        count = -1
                        captura.release()
                        break

                    if count > 5 and count < 9:
                        resize_image = cv2.resize(frame, resize_ratio)
			aux_vect_train = np.ravel(resize_image)
                        Xlist_train.append(aux_vect_train)

                        if video == '1.avi' or video == '2.avi':
                            ylist_train.append(0)
                            name_list.append(video_file)


                        else:
                            ylist_train.append(1)
                            name_list.append(video_file)


                print('se ha acabdao de leer el video %s' % video)
    print(aux_vect_train)
    print('SE HA ACABADO CON LOS VIDEOS DE TRAIN Y SE VA A COMENZAR CON LOS DE TEST')

    # Folder users are looked for in test folder
    for poss, user in enumerate(os.listdir(path_test)):
        image_folder_rgb = os.path.join(path_test, user)
        print('se van a leer los de usuario %s' % user)
        count = -1

        # Videos are looked for in users folder
        for pos2, video in enumerate(os.listdir(image_folder_rgb)):  # pos da la clase

            if video == 'HR_1.avi' or video == 'HR_2.avi' or video == 'HR_3.avi' or video == 'HR_4.avi':
                continue
            else:

                print('se va a leer el video %s' % video)
                video_file = os.path.join(image_folder_rgb, video)

                # Videos are read
                captura = cv2.VideoCapture(video_file)
                while (captura.isOpened()):
                    # For each frame of each video..
                    ret, frame = captura.read()
                    if ret == True:
                        count = count + 1
                    else:
                        count = -1
                        captura.release()
                        break

                    if count > 5 and count < 9:
                        resize_image = cv2.resize(frame, resize_ratio)
                        aux_vect_test = np.ravel(resize_image)
                        Xlist_test.append(aux_vect_test)

                        if video == '1.avi' or video == '2.avi':
                            ylist_test.append(0)
                            name_list.append(video_file)


                        else:
                            ylist_test.append(1)
                            name_list.append(video_file)

                print('se ha acabado de leer el video %s' % video)

    assert len(Xlist_test) == len(ylist_test)
    assert len(Xlist_train) == len(ylist_train)



    # NUmpy for training
    X_train = np.zeros([len(ylist_train), len(aux_vect_train)])
    y_train = np.zeros([len(ylist_train), ])

    for position, value in enumerate(Xlist_train):
        X_train[position] = value

    for position, value in enumerate(ylist_train):
        y_train[position] = value

    # Numpy for testing
    X_test = np.zeros([len(ylist_test), len(aux_vect_test)])
    y_test = np.zeros([len(ylist_test), ])

    for position, value in enumerate(Xlist_test):
        X_test[position] = value

    for position, value in enumerate(ylist_test):
        y_test[position] = value

    train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate = randomize_train(X_train, y_train, X_test,
                                                                                      y_test, 0.3, len(aux_vect_test))

    data_all = [train_index, test_index, validate_index, name_list,X_train, X_test, y_train, y_test, X_validate, y_validate]

    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()

    return train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate


def read_images_fiveScales(path_in, spacial_scales, path_out):

    test_ext = ('.JPG')
    X_list = []
    y_list = []

    countx = 0
    county = 0

    for pos, filename in enumerate(os.listdir(path_in)):
        image_folder = os.path.join(path_in, filename)

        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)

            aux_x = cv2.imread(image_path)

            aux_vecct_X = procesing(aux_x, spacial_scales)
            #
            # if aux_vecct_X != 0:

            for x in aux_vecct_X:
                    X_list.append(x)
                    countx+=1

                    # The outout of processing gives back 5 faces, so it is needed to add five labels
                    if filename == 'Real' or filename == 'users':
                        y_list.append(0)
                        county+=1

                    else:
                        y_list.append(1)
                        county+=1

    assert countx == county
    assert len(X_list) == len(y_list)

    X = np.zeros([len(y_list),49152])
    y = np.zeros([len(y_list),])

    for position, value in enumerate(X_list):
        X[position] = value

    for position, value in enumerate(y_list):
        y[position] = value

    train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate = Randomize_data(X, y, 0.2)

    data_all = [train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate]
    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()

    return train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate

def read_imagesOriginal(path_in, resize_ratio, path_out):
    #In this function images are going to be read and they are not cropped, just resized to the ratio choosen bu user
    X_list = []
    y_list = []

    countx = 0
    county = 0
    name_list = []

    for pos, filename in enumerate(os.listdir(path_in)):
        image_folder = os.path.join(path_in, filename)

        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)

            #Read image and resize it
            aux_x = cv2.imread(image_path)
            resize_image = cv2.resize(aux_x, resize_ratio)
            aux_vect = np.ravel(resize_image)
            X_list.append(aux_vect)
            countx += 1

            # Set label
            if filename == 'Real' or filename == 'users':
                y_list.append(0)
                county += 1
                name_list.append(image_path)


            else:
                y_list.append(1)
                county += 1
                name_list.append(image_path)


    len_image = len(aux_vect)
    assert countx == county
    assert len(X_list) == len(y_list)

    X = np.zeros([len(y_list), len_image])
    y = np.zeros([len(y_list), ])

    for position, value in enumerate(X_list):
        X[position] = value

    for position, value in enumerate(y_list):
        y[position] = value

    train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate = Randomize_data(
        X, y, 0.2, len_image)

    data_all = [train_index, test_index, validate_index, name_list,X_train, X_test, y_train, y_test, X_validate, y_validate]
    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()

    return train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate

def read_images_five_scales_all(path_in, spacial_scales, path_out):

    test_ext = ('.JPG')
    X_list = []
    y_list = []

    countx = 0
    county = 0

    for pos, filename in enumerate(os.listdir(path_in)):
        image_folder = os.path.join(path_in, filename)

        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)

            aux_x = cv2.imread(image_path)

            aux_vecct_X = procesings_lenfaces(aux_x, spacial_scales)
            #
            if aux_vecct_X is 0:
                print('this face does not have face:', image_path)
            else:
                for x in aux_vecct_X:
                    X_list.append(x)
                    countx += 1

                    # The outout of processing gives back 5 faces, so it is needed to add five labels
                    if filename == 'Real' or filename == 'users':
                        y_list.append(0)
                        county += 1

                    else:
                        y_list.append(1)
                        county += 1

    assert countx == county
    assert len(X_list) == len(y_list)

    X = np.zeros([len(y_list), 49152])
    y = np.zeros([len(y_list), ])

    for position, value in enumerate(X_list):
        X[position] = value

    for position, value in enumerate(y_list):
        y[position] = value

    train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate = Randomize_data(
        X, y, 0.2, len(x))

    data_all = [train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate]
    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()

    return train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate


def createFRAV_NIR(path_in, resize_ratio, path_out):
    path_RGB = path_in + '/RGB'
    path_NIR = path_in + '/NIR'

    countx = 0
    county = 0

    x = []
    y = []
    name_list = []
    for pos, filename in enumerate(os.listdir(path_RGB)):
        folder_RGB = os.path.join(path_RGB, filename)
        for image in os.listdir(folder_RGB):

            # Read images
            RGBimage_path = os.path.join(folder_RGB, image)
            folder_NIR = os.path.join(path_NIR, filename)
            image_NIR = (image.rsplit('.')[0])
            NIRimage_path = os.path.join(folder_NIR, image_NIR+'.jpg')

            aux_rgb = cv2.imread(RGBimage_path)
            aux_rgb = cv2.resize(aux_rgb, resize_ratio)
            aux_nir = cv2.imread(NIRimage_path,0)
            aux_nir = cv2.resize(aux_nir, resize_ratio)

            #Concatenate them
            image_contatenated = cv2.merge((aux_rgb, aux_nir))
            aux_image = np.ravel(image_contatenated)

            #append image
            x.append(aux_image)
            countx += 1

            # Set label
            if filename == 'Real' or filename == 'users':
                y.append(0)
                county += 1
                name_list.append(RGBimage_path)

            else:
                y.append(1)
                county += 1
                name_list.append(RGBimage_path)


    len_image = len(aux_image)

    assert countx == county
    assert len(x) == len(y)

    X = np.zeros([len(y), len_image])
    Y = np.zeros([len(y), ])

    for position, value in enumerate(x):
        X[position] = value

    for position, value in enumerate(y):
        Y[position] = value

    train_index, test_index, validate_index, X_train, X_test, y_train, y_test, X_validate, y_validate = Randomize_data(
        X, Y, 0.2, len_image)

    data_all = [train_index, test_index, validate_index, name_list,X_train, X_test, y_train, y_test, X_validate, y_validate]

    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()



def createFRAV_NIR_concatenate(path_in, resize_ratio, path_out):
    # to save rgb and nir separately but with images shuffeled in the same order.

    path_RGB = path_in + '/RGB'
    path_NIR = path_in + '/NIR'

    countx = 0
    county = 0

    x_rgb = []
    x_nir = []

    y = []

    name_list = []


    for pos, filename in enumerate(os.listdir(path_RGB)):
        folder_RGB = os.path.join(path_RGB, filename)
        for image in os.listdir(folder_RGB):

            # Read images
            RGBimage_path = os.path.join(folder_RGB, image)
            folder_NIR = os.path.join(path_NIR, filename)
            image_NIR = (image.rsplit('.')[0])
            NIRimage_path = os.path.join(folder_NIR, image_NIR+'.jpg')

            aux_rgb = cv2.imread(RGBimage_path)
            aux_rgb = cv2.resize(aux_rgb, resize_ratio)
            aux_nir = cv2.imread(NIRimage_path,0)
            aux_nir = cv2.resize(aux_nir, resize_ratio)

            x_rgb.append(np.ravel(aux_rgb))
            x_nir.append(np.ravel(aux_nir))

            countx += 1

            # Set label
            if filename == 'Real' or filename == 'users':
                y.append(0)
                county += 1
                name_list.append(RGBimage_path)

            else:
                y.append(1)
                county += 1
                name_list.append(RGBimage_path)


    len_image_rgb = len(np.ravel(aux_rgb))
    len_image_nir = len(np.ravel(aux_nir))


    assert countx == county
    assert len(x_rgb) == len(y)

    X_rgb = np.zeros([len(y), len_image_rgb])
    X_nir = np.zeros([len(y), len_image_nir])

    Y = np.zeros([len(y), ])

    for position, value in enumerate(x_rgb):
        X_rgb[position] = value
        X_nir[position] = x_nir[position]


    for position, value in enumerate(y):
        Y[position] = value

    assert len(X_rgb) == len(X_nir)

    train_index, test_index, validate_index, X_train_rgb, X_test_rgb, X_validate_rgb, X_train_nir, X_test_nir, X_validate_nir, y_train, y_test, y_validate = Randomize_data_rgb_nir_concatenated(
        X_rgb, X_nir, Y, 0.2, len_image_rgb, len_image_nir)

    data_all = [train_index, test_index, validate_index, name_list,X_train_rgb, X_test_rgb, X_validate_rgb, X_train_nir, X_test_nir, X_validate_nir, y_train,  y_test, y_validate]

    save_file2 = open(path_out, 'wb')
    pickle.dump(data_all, save_file2, -1)
    save_file2.close()

def generar_genuine_attack(path_in, resize_ratio, path_out):

    genuine_list = []
    genuine_name_list = []
    attack_list = []
    attack_name_list = []

    # Read folder
    for pos, filename in enumerate(os.listdir(path_in)):
        image_folder = os.path.join(path_in, filename)

        #Read image
        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)

            # Read image and resize it
            aux_x = cv2.imread(image_path)
            resize_image = cv2.resize(aux_x, resize_ratio)
            aux_vect = np.ravel(resize_image)
            # Set label
            if filename == 'Real' or filename == 'users':
                genuine_list.append((aux_vect))
                genuine_name_list.append(image_path)

            else:
                attack_list.append((aux_vect))
                attack_name_list.append(image_path)

    assert(len(attack_list) == len(attack_name_list))
    assert(len(genuine_list) == len(genuine_name_list))

    gen_list = np.zeros([len(genuine_name_list), len(aux_vect)])
    att_list = np.zeros([len(attack_name_list), len(aux_vect)])

    for position, value in enumerate(genuine_list):
        gen_list[position] = value

    for position, value in enumerate(attack_list):
        att_list[position] = value

    data = [gen_list, genuine_name_list, att_list, attack_name_list]
    print(gen_list)
    print(att_list)
    save_file2 = open(path_out, 'wb')
    pickle.dump(data, save_file2, -1)
    save_file2.close()


def generar_genuine_attack_multispectral(path_in, resize_ratio, path_out):

    genuine_list_nir = []
    genuine_name_list_nir = []
    attack_list_nir = []
    attack_name_list_nir = []

    genuine_list_rgb = []
    genuine_name_list_rgb = []
    attack_list_rgb = []
    attack_name_list_rgb = []

    genuine_list_thr = []
    genuine_name_list_thr = []
    attack_list_thr = []
    attack_name_list_thr = []

    genuine_list_merged = []
    attack_list_merged = []

    THR_dir = path_in # +'/THERMAL_GRAY'
    # Read folder
    for pos, filename in enumerate(os.listdir(THR_dir)):
        image_folder = os.path.join(THR_dir, filename)

        #Read image
        for image in os.listdir(image_folder):

            # Read NIR image and resize it
            if filename == 'attack_04':
                aux_nir=cv2.imread(path_in+'/NIR/'+filename+'/frame_0.jpg',0)

            else:
                aux_nir=cv2.imread(path_in+'/NIR/'+filename+'/'+image,0)

            resize_image_nir = cv2.resize(aux_nir, resize_ratio)
            aux_vect_nir = np.ravel(resize_image_nir)

            # Read RGB image and resize it
            rgb_path_image = path_in+'/RGB/'+filename+'/'+image.split('.')[0]+'.JPG'
            aux_rgb = cv2.imread(rgb_path_image)
            resize_image_rgb = cv2.resize(aux_rgb, resize_ratio)
            aux_vect_rgb = np.ravel(resize_image_rgb)

            # Read THR image and resize it
            image_path = os.path.join(image_folder, image)
            aux_thr = cv2.imread(image_path,0)
            resize_image_thr = cv2.resize(aux_thr, resize_ratio)
            aux_vect_thr = np.ravel(resize_image_thr)


            # Set label
            if filename == 'Real' or filename == 'users':
                genuine_list_nir.append((aux_vect_nir))
                genuine_name_list_nir.append(image_path)

                genuine_list_rgb.append((aux_vect_rgb))
                genuine_name_list_rgb.append(path_in+'/RGB/'+image)

                genuine_list_thr.append((aux_vect_thr))
                genuine_name_list_thr.append(path_in+'/THR/'+image)

                aux_merged = np.ravel(cv2.merge((resize_image_rgb, resize_image_nir, resize_image_thr)))
                genuine_list_merged.append(aux_merged)

            else:
                attack_list_nir.append((aux_vect_nir))
                attack_name_list_nir.append(image_path)

                attack_list_rgb.append((aux_vect_rgb))
                attack_name_list_rgb.append(path_in+'/RGB/'+image)

                attack_list_thr.append((aux_vect_thr))
                attack_name_list_thr.append(path_in+'/THR/'+image)

                attack_list_merged.append(np.ravel(cv2.merge((resize_image_rgb, resize_image_nir, resize_image_thr))))

    assert(len(attack_list_rgb) == len(attack_name_list_rgb))
    assert(len(genuine_list_rgb) == len(genuine_name_list_rgb))
    assert(len(attack_list_rgb) == len(attack_name_list_nir) == len(attack_name_list_thr))

    gen_list_nir = np.zeros([len(genuine_name_list_nir), len(aux_vect_nir)])
    att_list_nir = np.zeros([len(attack_name_list_nir), len(aux_vect_nir)])
    gen_list_rgb = np.zeros([len(genuine_name_list_rgb), len(aux_vect_rgb)])
    att_list_rgb = np.zeros([len(attack_name_list_rgb), len(aux_vect_rgb)])
    gen_list_thr = np.zeros([len(genuine_name_list_thr), len(aux_vect_nir)])
    att_list_thr = np.zeros([len(attack_name_list_thr), len(aux_vect_nir)])
    gen_list_merged = np.zeros([len(genuine_name_list_thr), len(aux_merged)])
    att_list_merged = np.zeros([len(attack_name_list_thr), len(aux_merged)])

    for position, value in enumerate(genuine_list_nir):
        gen_list_nir[position] = value
        gen_list_rgb[position] = genuine_list_rgb[position]
        gen_list_thr[position] = genuine_list_thr[position]
        gen_list_merged[position] = genuine_list_merged[position]

    for position, value in enumerate(attack_list_nir):
        att_list_nir[position] = value
        att_list_rgb[position] = attack_list_rgb[position]
        att_list_thr[position] = attack_list_thr[position]
        att_list_merged[position] = attack_list_merged[position]

    data = [[gen_list_rgb, genuine_name_list_rgb, att_list_rgb, attack_name_list_rgb],[gen_list_nir, genuine_name_list_nir, att_list_nir, attack_name_list_nir],[gen_list_thr, genuine_name_list_thr, att_list_thr, attack_name_list_thr]]


    data2 = [gen_list_merged, genuine_name_list_rgb, att_list_merged, attack_name_list_rgb]

    save_file2 = open(path_out+'_separated.pkl', 'wb')
    pickle.dump(data, save_file2, -1)
    save_file2.close()


    save_file2 = open(path_out+'_merged.pkl', 'wb')
    pickle.dump(data2, save_file2, -1)
    save_file2.close()

if __name__ == '__main__':
    # path_in =  'databases/from'
    # spacial_scales = [1, 1.4, 1.8, 2.2, 2.6]
    # spacial_scales = [1]
    # resize_ratio = (128,128)
    # resize_ratio = (1,1)
    # path_out = 'data_frav_128x128_2.pkl'
    # read_imagesOriginal(path_in, resize_ratio, path_out)
    #read_images_five_scales_all(path_in, spacial_scales, path_out)

    # path_in =  'databases/Casia database Fixed'
    # # spacial_scales = [1]
    # resize_ratio = (128,128)
    # # resize_ratio = (1,1)
    # path_out = 'casia_prueba.pkl'
    # read_imagesOriginal(path_in, resize_ratio, path_out)

    # path_in = 'databases/MFSD/mfsd'
    # path_out = 'mfsd_positivas_negativas_128_128.pkl'

    resize_ratio = (128,128)
    # generar_genuine_attack(path_in, resize_ratio, path_out)

    # path_in = 'databases/Casia database Fixed'
    # path_out = 'casia_positivas_negativas_128_128.pkl'
    path_in = 'databases/from'
    path_out = 'frav_multispectral_positivas_negativas_128_128'
    generar_genuine_attack_multispectral(path_in, resize_ratio, path_out)

    #  path_in = 'databases/cbsr_antispoofing'
    # path_in = '/home/beaa/Escritorio/Theano/DATABASES/cbsr_antispoofing'
    # spacial_scales = [1]
    # resize_ratio = (128,128)
    # resize_ratio = (1,1)
    # path_out = 'data_casia_vid_128x128_2.pkl'
    # read_videos2(path_in, resize_ratio, path_out)

    # path_in =  'databases/RGB_NIR'
    # spacial_scales = [1, 1.4, 1.8, 2.2, 2.6]
    # spacial_scales = [1]
    # resize_ratio = (128,128)
    # # resize_ratio = (1,1)
    # path_out = 'data_frav_RGB_NIR_concatenate_128x128_2.pkl'
    # createFRAV_NIR_concatenate(path_in, resize_ratio, path_out)
    # #read_images_five_scales_all(path_in, spacial_scales, path_out)

    # path_out = 'data_frav_RGB_NIR_128x128.pkl_2.pkl'
    # createFRAV_NIR(path_in, resize_ratio, path_out)
