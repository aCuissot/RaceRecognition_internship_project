# coding: utf-8
import csv
import pickle

# import dlib
import cv2
import numpy as np

# Loading csv into labs

labs = []
with open('wiki_imdb_margine_40.csv', newline='') as csvfile:
    labels2 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels2:
        labs.append(row)

# Using relative path
# labs[n][2] means taking the 2nd cell of the row or rather the path
# some csv (like Chalearn) don't have gender information, so they have 2 cells per line and not 3

filenames = []
leng = len(labs)
for n in range(0, leng):
    filenames.append(labs[n][2])

filenames

# ONLY FOR WIKI-IMDB

# Gender/Age division
# creating indexes that I will need to identify male / female and age group.

p = 'db/img40/'
# Vectors for images
imgf = []
imgm = []

# For labels
labf = np.zeros((5000, 2))
labm = np.zeros((5000, 2))

# Indices to help with the procedure
f014 = 0
m014 = 0
f1525 = 0
m1525 = 0
f2639 = 0
m2639 = 0
f4055 = 0
m4055 = 0
f56 = 0
m56 = 0
pos = 0
pos1 = 0
pos2 = 0

shuffle_index = np.random.permutation(len(labs))

i = 0

while (pos1 + pos2) != 10000 and i < len(labs):
    if labs[shuffle_index[i]][0] != 'NaN':
        gender = int(labs[shuffle_index[i]][0])
        age = int(labs[shuffle_index[i]][1])
        path = p + labs[shuffle_index[i]][2]
        if gender == 0 and age < 15 and f014 < 1000:
            imgf.append(cv2.imread(path))
            labf[pos1][0] = gender
            labf[pos1][1] = age
            f014 = f014 + 1
            pos1 = pos1 + 1
        elif gender == 1 and age < 15 and m014 < 1000:
            imgm.append(cv2.imread(path))
            labm[pos2][0] = gender
            labm[pos2][1] = age
            m014 = m014 + 1
            pos2 = pos2 + 1
        elif gender == 0 and 14 < age < 26 and f1525 < 1000:
            imgf.append(cv2.imread(path))
            labf[pos1][0] = gender
            labf[pos1][1] = age
            f1525 = f1525 + 1
            pos1 = pos1 + 1
        elif gender == 1 and 14 < age < 26 and m1525 < 1000:
            imgm.append(cv2.imread(path))
            labm[pos2][0] = gender
            labm[pos2][1] = age
            m1525 = m1525 + 1
            pos2 = pos2 + 1
        elif gender == 0 and 25 < age < 40 and f2639 < 1000:
            imgf.append(cv2.imread(path))
            labf[pos1][0] = gender
            labf[pos1][1] = age
            f2639 = f2639 + 1
            pos1 = pos1 + 1
        elif gender == 1 and 25 < age < 40 and m2639 < 1000:
            imgm.append(cv2.imread(path))
            labm[pos2][0] = gender
            labm[pos2][1] = age
            m2639 = m2639 + 1
            pos2 = pos2 + 1
        elif gender == 0 and 39 < age < 56 and f4055 < 1000:
            imgf.append(cv2.imread(path))
            labf[pos1][0] = gender
            labf[pos1][1] = age
            f4055 = f4055 + 1
            pos1 = pos1 + 1
        elif gender == 1 and 39 < age < 56 and m4055 < 1000:
            imgm.append(cv2.imread(path))
            labm[pos2][0] = gender
            labm[pos2][1] = age
            m4055 = m4055 + 1
            pos2 = pos2 + 1
        elif gender == 0 and age > 55 and f56 < 1000:
            imgf.append(cv2.imread(path))
            labf[pos1][0] = gender
            labf[pos1][1] = age
            f56 = f56 + 1
            pos1 = pos1 + 1
        elif gender == 1 and age > 55 and m56 < 1000:
            imgm.append(cv2.imread(path))
            labm[pos2][0] = gender
            labm[pos2][1] = age
            m56 = m56 + 1
            pos2 = pos2 + 1
    i = i + 1

print("finish")
print(m014, m1525, m2639, m4055, m56, f014, f1525, f2639, f4055, f56)
# Having imgm, imgf, labm, labf.
# training, test and validation sequences


# Vector X for images and y per label, balancing gender

y = np.zeros((10000, 2))

X = imgm + imgf

for i in range(0, 5000):
    y[i][0] = labm[i][0]
    y[i][1] = labm[i][1]
k = 0
for i in range(5000, 10000):
    y[i][0] = labf[k][0]
    y[i][1] = labf[k][1]
    k = k + 1

X

# Some parameters for image crop
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

# calc hog

hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
X_hog = []

for images in X:
    gray_img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    faceimg = gray_img[ny:ny + nr, nx:nx + nr]
    resized = cv2.resize(faceimg, (128, 128))
    h = hog.compute(resized)
    hogghy = np.zeros(1764)
    for i in range(0, len(h)):
        hogghy[i] = h[i][0]
    X_hog.append(hogghy)

# saving weight

with open("namehog.pkl", "wb") as output:
    pickle.dump(X_hog, output)

with open("namelabel.pkl", "wb") as output:
    pickle.dump(y, output)

# EXAMPLE WITH WIKI IMDB


# from sklearn_deap_master.evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import svm

with open("namehog.pkl", "rb") as data:
    Hog = pickle.load(data)
with open("namelabel.pkl", "rb") as data:
    y = pickle.load(data)

Hog = X_hog

# Create 3 sets for Hog: Train, validation and test. Create a label vector Y too

X_train_hog = []
X_test_hog = []
X_vali_hog = []
Y = y

# Entries work randomly
# To divide into males and females I use 2 other vectors. Remember that the first 5000 of the Hog are male, the others female.
X_man = []
X_woman = []

for i in range(0, 5000):
    Y[i] = y[i]
    X_man.append(Hog[i])

for i in range(5000, 10000):
    Y[i] = y[i]
    X_woman.append(Hog[i])

# Balancing gender
X_train_hog = X_man[:3500] + X_woman[:3500]
X_test_hog = X_man[3500:4250] + X_woman[3500:4250]
X_vali_hog = X_man[4250:] + X_woman[4250:]

# Here I deal with labels in the same way. 1 stands for gender label (I create gender vectors only).

y_train_1 = np.zeros(7000)
y_test_1 = np.zeros(1500)
y_vali_1 = np.zeros(1500)
k = 0
for i in range(0, 3500):
    y_train_1[i] = Y[k][0]
    k = k + 1
for i in range(0, 750):
    y_test_1[i] = Y[k][0]
    k = k + 1
for i in range(0, 750):
    y_vali_1[i] = Y[k][0]
    k = k + 1
for i in range(3500, 7000):
    y_train_1[i] = Y[k][0]
    k = k + 1
for i in range(750, 1500):
    y_test_1[i] = Y[k][0]
    k = k + 1
for i in range(750, 1500):
    y_vali_1[i] = Y[k][0]
    k = k + 1

# Training.
# As I already have the parameters, I could combine train and validation to get 8500 samples per fit.
svc = svm.SVC(kernel='linear', C=0.17782)
svc.fit(X_train_hog + X_vali_hog, np.append(y_train_1, y_vali_1))

# Validation

test = svc.predict(X_vali_hog)
print(test)
c_m = confusion_matrix(y_vali_1, test)
print(c_m)
mscore = svc.score(X_vali_hog, y_vali_1)
print(mscore)

# Calc score and test

test = svc.predict(X_train_hog + X_vali_hog)
print(test)

np.set_printoptions(threshold=np.nan)
print(test)

mscore = svc.score(X_train_hog + X_vali_hog, np.append(y_train_1, y_vali_1))
print(mscore)

m = 0
fem = 0
truth = np.append(y_train_1, y_vali_1)
for i in range(0, len(test)):
    if truth[i] == 0 and truth[i] == test[i]:
        fem = fem + 1
    if truth[i] == 1 and truth[i] == test[i]:
        m = m + 1
print(fem / 4250, m / 4250)

c_m = confusion_matrix(np.append(y_train_1, y_vali_1), test)
print(c_m)

# Saving classifier with joblib

from sklearn.externals import joblib

joblib.dump(svc, 'namesvc.pkl')
