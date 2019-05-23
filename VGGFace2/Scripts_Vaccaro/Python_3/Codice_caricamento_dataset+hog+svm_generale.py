# coding: utf-8


import csv
import pickle

import cv2
import numpy as np

# Loading cvs file in 'labs'

labs = []
with open('nomecsv.csv', newline='') as csvfile:
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

# These are parameters for "cutting" an image with a 40% margin.
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

# create hog

# defining hog
hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)

# support parameters
X_hog = []
y = []
n = 0
img = None
gray_img = None
h = None
hogghy = np.zeros(1764)

# From the relative path: load image, grayscale it, cut it, resize it to 128*128 and calculate hog
for images in filenames:
    img = cv2.imread(images)

    if img is not None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceimg = gray_img[ny:ny + nr, nx:nx + nr]
        resized = cv2.resize(faceimg, (128, 128))

        h = hog.compute(resized)

        for i in range(0, len(h)):
            hogghy[i] = h[i][0]

        X_hog.append(hogghy)

        # Note that here I take only the gender (the first cell of the line)
        y.append(int(labs[n][0]))
    n = n + 1

# Saving hog and labels in pkl file

# with open("hog.pkl","wb") as output:
#        pickle.dump(X_hog,output)
# with open("etichetta.pkl","wb") as output:
#    pickle.dump(y,output)


# Save the weight in a pkl.

with open("namehog.pkl", "wb") as output:
    pickle.dump(X_hog, output)

with open("namelabel.pkl", "wb") as output:
    pickle.dump(y, output)

# AN EXAMPLE OF TRAINING WITH VGG


# from sklearn_deap_master.evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn import svm

with open("namehog.pkl", "rb") as data:
    Hog = pickle.load(data)
with open("namelabel.pkl", "rb") as data:
    y = pickle.load(data)

# Initiate vectors (useless here)

X_train_hog = Hog

y_train_1 = np.zeros(len(y))
k = 0
for i in range(0, len(y)):
    y_train_1[i] = y[k]
    k = k + 1

# Case of labels with comma
Y_train_1 = []
for n in range(0, len(y_train_1)):
    Y_train_1.append(int(y_train_1[n]))

print("-----")

# Train.

svc = svm.SVC(kernel='linear', C=0.17782, verbose=True)
svc.fit(X_train_hog, Y_train_1)

# saving calssifier with joblib

from sklearn.externals import joblib

joblib.dump(svc, 'svc_vgg_linear.pkl')

# calc score

mscore = svc.score(X_train_hog, Y_train_1)
print(mscore)
