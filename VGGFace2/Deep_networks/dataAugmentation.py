import random
import numpy as np
import cv2
import csv
import keras

img_size_row = 224
img_size_column = 224
global IMAGE_SIZE
IMAGE_SIZE = (img_size_row, img_size_column)
num_classes = 1


def onlinedataaug(img):
    # if they all come out 0, no data augmentation, otherwise sum of the effects of the 1 coming out
    if random.randint(0, 1):  # gauss
        # print("gauss")
        row, col, ch = img.shape

        binom = np.random.binomial(n=500, p=0.5, size=(row, col, ch))
        binom = binom - 250

        noisy = img + binom

        for i in range(0, row):
            for j in range(0, col):
                for l in range(0, ch):
                    if noisy[i][j][l] < 0:
                        noisy[i][j][l] = 0
                    elif noisy[i][j][l] > 255:
                        noisy[i][j][l] = 255

        img = noisy.astype(np.uint8)

    if random.randint(0, 1):  # resize
        # print("resize")
        row, col, ch = img.shape
        p = random.randint(1, 2) * 2
        row = row / p
        col = col / p
        img = cv2.resize(img, (int(row), int(col)), 0, 0, cv2.INTER_LINEAR)

    # if random.randint(0,1):  # brightness  alternately adding and subtracting lightness
    # print("brightness")
    #    exposition = random.randint(1,2) #se 1 overexposure se 2 underexposure
    #    if exposition==2:
    #        param=random.randint(1,2)/10
    #        param = 0 - param

    #    else :
    #        param=random.randint(2,3)/10

    #    br= tf.image.adjust_brightness(immagine,param)
    #    with tf.Session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        immagine = sess.run(br)
    #        sess.close()

    # if random.randint(0,1): #flip
    #    #print("flip")
    #    imgflip = tf.image.flip_left_right(immagine)
    #    with tf.Session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        immagine = sess.run(imgflip)
    #        sess.close()

    if random.randint(0, 1):  # brightess: alternately I add and subtract lighting
        # print("brightness")

        exposition = random.randint(1, 2)  # 1 = overexposed, 2 = underexposed
        if exposition == 2:
            brig = img * 0.7

        else:
            brig = img * 1.4

        row, col, ch = brig.shape
        for i in range(0, row):
            for j in range(0, col):
                for k in range(0, ch):
                    if brig[i][j][k] > 255:
                        brig[i][j][k] = 255
                    elif brig[i][j][k] < 0:
                        brig[i][j][k] = 0
        img = brig.astype(np.uint8)

    if random.randint(0, 1):  # flip
        # print("flip")
        flip = np.fliplr(img)
        img = flip

    return img


def data_generator(csvpath):
    labs = []
    with open(csvpath, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels:
            labs.append(row)

    p = np.array(labs)
    np.random.shuffle(p)

    for row in p:
        path = row[2]
        image = cv2.imread(path)

        im = np.array(image)
        gender = np.array(int(row[0]))
        age = np.array(int(row[1]))

        yield (im, gender, age)


def data_generator_prepr(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        age_list = []
        for i in data:
            # adding same definition of ny, nx, nr than bellow, hoping that s good
            x = 51
            yy = 51
            w = 154
            h = 154
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = yy + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            immagine = i[0]

            immagine = immagine[ny:ny + nr, nx:nx + nr]

            image = cv2.resize(immagine, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)
            image /= 255

            img_list.append(image)
            gender_list.append(i[1])
            age_list.append(i[2])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)

                # age=age_list.copy()
                # gender=np.array(gender_categorical)
                age = np.array(age_list)
                # labels=np.array([gender,[age]])
                img_list = []
                gender_list = []
                age_list = []
                yield (im_p, [gender_categorical, age])


def data_generator_con_data_aug(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        age_list = []
        for i in data:

            immagine = i[0]

            x = 51
            yy = 51
            w = 154
            h = 154
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = yy + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            # random crop con prob + alta su 0 (gaussiana)
            sigma = 8
            mean = 0
            gauss = np.random.normal(mean, sigma, (2, 1))

            gauss = gauss.astype(int)
            if gauss[0] > 8:
                gauss[0] = 8
            elif gauss[0] < -8:
                gauss[0] = -8
            if gauss[1] > 8:
                gauss[1] = 8
            elif gauss[1] < -8:
                gauss[1] = -8

            ny = ny + gauss[0][0]
            nx = nx + gauss[1][0]

            immagine = immagine[ny:ny + nr, nx:nx + nr]

            image = onlinedataaug(immagine)

            image = cv2.resize(image, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)
            image /= 255

            img_list.append(image)
            gender_list.append(i[1])
            age_list.append(i[2])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)

                age = np.array(age_list)
                img_list = []
                gender_list = []
                age_list = []
                yield (im_p, [gender_categorical, age])
