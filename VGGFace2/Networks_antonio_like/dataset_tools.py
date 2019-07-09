import numpy as np
import random
import keras
import os
from glob import glob
import cv2

NUM_CLASSES = 4


def load_image(impath, shape, image_process_fcn=None):
    input_img = cv2.imread(impath)
    if input_img is None:
        import sys
        print("Unable to load image file %s" % impath)
        return None  # sys.exit(1)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (shape[1], shape[2]), 0, 0, cv2.INTER_LINEAR)
    if image_process_fcn is None:
        pass
    else:
        input_img = image_process_fcn(input_img)
    input_img = cv2.resize(input_img, (shape[1], shape[2]), 0, 0, cv2.INTER_LINEAR)

    input_img = input_img.astype(np.float32)
    input_img /= 255
    input_img = input_img.reshape(shape)
    return input_img


def load_dataset(dirpath, shape, doShuffle=False, image_process_fcn=None):
    training_distr = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    data = [None, None, None, None]
    data[0] = glob(os.path.join(dirpath, '0', '*.jpg')) + glob(os.path.join(dirpath, '0', '*.png'))
    data[1] = glob(os.path.join(dirpath, '1', '*.jpg')) + glob(os.path.join(dirpath, '1', '*.png'))
    data[2] = glob(os.path.join(dirpath, '2', '*.jpg')) + glob(os.path.join(dirpath, '2', '*.png'))
    data[3] = glob(os.path.join(dirpath, '3', '*.jpg')) + glob(os.path.join(dirpath, '3', '*.png'))
    n = [0, 0, 0, 0]
    print("African American: %d, Asian: %d, Latin: %d, Indian: %d" % (
        len(data[0]), len(data[1]), len(data[2]), len(data[3])))
    num = len(data[0]) + len(data[1]) + len(data[2]) + len(data[3])
    print(num)

    if doShuffle:
        np.random.shuffle(data[0])
        np.random.shuffle(data[1])
        np.random.shuffle(data[2])
        np.random.shuffle(data[3])
    for i in range(num):
        # label = i%NUM_CLASSES
        label = training_distr[i % 64]
        # if doShuffle:
        #    label = random.randint(0,NUM_CLASSES-2)
        # path = data[label][n[label]]
        # n[label]+=1
        try:
            path = data[label][n[label]]
            n[label] += 1
        except:
            break
        # print(label)
        image = load_image(path, shape, image_process_fcn)
        if image is None:
            continue

        if False and shape[1] == 96:
            vis_img = image
            vis_img *= 255
            vis_img = vis_img.astype(np.uint8)
            # print((shape[1],shape[2],shape[3]))
            vis_img = vis_img.reshape((shape[1], shape[2], shape[3]))
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("im %d" % label, vis_img)
            cv2.waitKey(500)

        im = np.array(image)
        yield (im, label)


def onlinedataaug(image):
    # Gaussian noise
    if random.randint(0, 1):
        row, col, ch = image.shape
        binom = np.random.binomial(n=500, p=0.5, size=(row, col, ch))
        binom = binom - 250
        noisy = image + binom
        noisy = np.clip(noisy, 0, 255)
        image = noisy.astype(np.uint8)
    # Resize
    if random.randint(0, 1):
        row, col, ch = image.shape
        p = random.randint(1, 2) * 2
        row = row / p
        col = col / p
        image = cv2.resize(image, (int(row), int(col)), 0, 0, cv2.INTER_LINEAR)
    # brightess : alternatively adding and subtracting lighting
    if random.randint(0, 1):
        esposizione = random.randint(1, 2)  # se 1 overexpose se 2 underexpose
        if esposizione == 2:
            brigh = image * 0.7
        else:
            brigh = image * 1.4
        brigh = np.clip(brigh, 0, 255)
        image = brigh.astype(np.uint8)
    # flip
    if random.randint(0, 1):
        flip = np.fliplr(image)
        image = flip

    return image


def load_dataset_for_pred(dirpath, shape, doShuffle=False, image_process_fcn=None):
    data = [None, None, None, None]
    data[0] = glob(os.path.join(dirpath, '0', '*.jpg')) + glob(os.path.join(dirpath, '0', '*.png'))
    data[1] = glob(os.path.join(dirpath, '1', '*.jpg')) + glob(os.path.join(dirpath, '1', '*.png'))
    data[2] = glob(os.path.join(dirpath, '2', '*.jpg')) + glob(os.path.join(dirpath, '2', '*.png'))
    data[3] = glob(os.path.join(dirpath, '3', '*.jpg')) + glob(os.path.join(dirpath, '3', '*.png'))
    n = [0, 0, 0, 0]
    print("African American: %d, Asian: %d, Latin: %d, Indian: %d" % (
        len(data[0]), len(data[1]), len(data[2]), len(data[3])))
    num = len(data[0]) + len(data[1]) + len(data[2]) + len(data[3])
    print(num)

    if doShuffle:
        np.random.shuffle(data[0])
        np.random.shuffle(data[1])
        np.random.shuffle(data[2])
        np.random.shuffle(data[3])
    for i in range(NUM_CLASSES):
        print(i)
        for j in range(len(data[i])):
            label = i
            try:
                path = data[label][n[label]]
                n[label] += 1
                # print(n)
            except:
                print("Except %s" % path)
                continue
            # print(label)
            image = load_image(path, shape, image_process_fcn)
            if image is None:
                print("Image %s not found" % path)
                continue

            if False and shape[1] == 96:
                vis_img = image
                vis_img *= 255
                vis_img = vis_img.astype(np.uint8)
                # print((shape[1],shape[2],shape[3]))
                vis_img = vis_img.reshape((shape[1], shape[2], shape[3]))
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("im %d" % label, vis_img)
                cv2.waitKey(500)

            im = np.array(image)
            yield (im, label)


def full_augmentation(image):
    siz = image.shape[0]
    # random crop with prob + stop on 0 (gaussian)
    sigma = 12
    mean = 0
    gauss = np.random.normal(mean, sigma, (4, 1))

    gauss = gauss.astype(int)
    for i in range(4):
        if gauss[i] > sigma:
            gauss[i] = sigma
        elif gauss[i] < 0:
            gauss[i] = 0

    image = image[gauss[0][0]:siz - gauss[1][0], gauss[2][0]:siz - gauss[3][0]]

    image = onlinedataaug(image)
    return image


def load_for_training(dirpath, batchsize, shape):
    img_list = []
    label_list = []
    while 1:
        ds = load_dataset(dirpath, shape, doShuffle=True, image_process_fcn=full_augmentation)
        for i in ds:
            img_list.append(i[0])
            label_list.append(i[1])
            if len(img_list) == batchsize:
                im_p = np.array(img_list).reshape(batchsize, shape[1], shape[2], shape[3])
                label_categorical = keras.utils.to_categorical(label_list, num_classes=NUM_CLASSES)
                img_list = []
                label_list = []
                yield (im_p, label_categorical)


def load_for_test(dirpath, batchsize, shape):
    img_list = []
    label_list = []
    while 1:
        ds = load_dataset(dirpath, shape)
        for i in ds:
            img_list.append(i[0])
            label_list.append(i[1])
            if len(img_list) == batchsize:
                im_p = np.array(img_list).reshape(batchsize, shape[1], shape[2], shape[3])
                label_categorical = keras.utils.to_categorical(label_list, num_classes=NUM_CLASSES)
                img_list = []
                label_list = []
                yield (im_p, label_categorical)


def load_for_pred(dirpath, batchsize, shape):
    img_list = []
    label_list = []
    ds = load_dataset(dirpath, shape)
    for i in ds:
        img_list.append(i[0])
        label_list.append(i[1])
    print(len(label_list))
    im_p = np.array(img_list).reshape(len(label_list), shape[1], shape[2], shape[3])
    label_categorical = keras.utils.to_categorical(label_list, num_classes=NUM_CLASSES)
    return im_p, label_categorical


def dataset_size(csv_path):
    ds = load_dataset(csv_path, (1, 64, 64, 3))
    count = 0
    for i in ds:
        count += 1
    return count


def evaluate_performance(dirpath, batchsize, shape, network):
    ds = load_dataset_for_pred(dirpath, shape)
    Y_pred = []
    Y_true = []
    for i in ds:
        lab = network.predict(i[0], batchsize, verbose=1)
        print(lab)
        Y_pred.append(np.argmax(lab))
        Y_true.append(i[1])
    return np.array(Y_pred), keras.utils.to_categorical(Y_true, num_classes=NUM_CLASSES)
