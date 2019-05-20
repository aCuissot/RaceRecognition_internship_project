# coding: utf-8


import random
import deap
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib as mt
import csv
import pickle
import sys
import cv2
import gc
import numpy as np
import os

# Hog contain hog and y contain label

with open("nome_hog.pkl", "rb") as data:
    Hog = pickle.load(data)

with open("nome_etichetta.pkl", "rb") as data:
    y = pickle.load(data)

# Lauching shuffle (optional)

c = list(zip(Hog, y))

random.shuffle(c)

Hog, y = zip(*c)

# Making train and validation sets
X_train_hog = []
X_vali_hog = []

y_train = np.zeros(14755)
y_vali = np.zeros(2755)

for i in range(0, 12000):
    y_train[i] = y[i]
    X_train_hog.append(Hog[i])
k = 0
for i in range(12000, 14755):
    y_vali[k] = y[i]
    X_vali_hog.append(Hog[i])
    k = k + 1

# Optional, only if we got a float instead of int

Y_train = []
for n in range(0, len(y_train)):
    Y_train.append(int(y_train[n]))
print(Y_train)

Y_vali = []
for n in range(0, len(y_vali)):
    Y_vali.append(int(y_vali[n]))
print(Y_vali)
print("-----")

# Free some memory
gc.collect()

# Create SVR and fit it

svr = svm.SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=4, verbose=True)
svr.fit(X_train_hog, Y_train)

# Here is the score

mscore = svr.score(X_vali_hog, Y_vali)
print(mscore)

# Printing test result and ground truth

test = svr.predict(X_vali_hog)
print(test)
print(Y_vali)


def middle_error(prevision, ground_truth):
    middle_error = 0
    for i in range(0, len(prevision)):
        error = (int(prevision[i]) - ground_truth[i])
        if error < 0:
            error = error * (-1)
        middle_error = middle_error + error
    middle_error = middle_error / len(prevision)
    return middle_error


def middle_error_per_age(prevision, ground_truth):
    middle_error = np.zeros((150, 3))
    for i in range(0, len(middle_error)):
        middle_error[i][0] = i + 1
        middle_error[i][1] = 0
        middle_error[i][2] = 0
    for i in range(0, len(prevision)):
        error = (int(prevision[i]) - ground_truth[i])
        if error < 0:
            error = error * (-1)
        if (ground_truth[i] > 0 and ground_truth[i] < len(middle_error)):
            middle_error[int(ground_truth[i]) - 1][1] += error
            middle_error[int(ground_truth[i]) - 1][2] += 1
    for i in range(0, len(middle_error)):
        if (middle_error[i][2] > 0 and middle_error[i][1] > 0):
            middle_error[i][1] = middle_error[i][1] / middle_error[i][2]
    result = [[0 for x in range(3)] for y in range(len(middle_error))]
    for i in range(0, len(middle_error)):
        result[i][0] = (int(middle_error[i][0]))
        result[i][1] = np.around(middle_error[i][1], decimals=2)
        result[i][2] = (int(middle_error[i][2]))
    return result


# Return the ages with error(age)>average(errors)

def biggest_errors(middle_error_age, mid):
    errors_array = [0] * len(middle_error_age)
    for i in range(0, len(middle_error_age)):
        errors_array[i] = middle_error_age[i][1]
    biggest_errors = [i for i, x in enumerate(errors_array) if x > mid]
    result = [[0 for x in range(3)] for y in range(len(biggest_errors))]
    for i in range(0, len(biggest_errors)):
        result[i] = middle_error_age[biggest_errors[i]]
    return result


# Function for bands division, division in 4 bands defined by A, B and C (here hardcoded)

def score_per_fasce(prevision, ground_truth):
    # Support parameters
    ok = 0
    minore = 0
    g_a = 0
    adulto = 0
    anziano = 0
    e_minore = 0
    e_g_a = 0
    e_adulto = 0
    e_anziano = 0

    # Band 1 : 0 <= x < A
    # Band 2 : A <= x < B
    # Band 3 : B <= x < C
    # Band 4 : C <= x
    A = 18
    B = 46
    C = 66

    for i in range(0, len(prevision)):

        if ground_truth[i] < A:
            truth = 'young'
        elif ground_truth[i] >= A and ground_truth[i] < B:
            truth = 'early adult'
        elif ground_truth[i] >= B and ground_truth[i] < C:
            truth = 'adult'
        elif ground_truth[i] >= C:
            truth = 'elder'

        if prevision[i] < A:
            test = 'young'
        elif prevision[i] >= A and prevision[i] < B:
            test = 'early adult'
        elif prevision[i] >= B and prevision[i] < C:
            test = 'adult'
        elif prevision[i] >= C:
            test = 'elder'

        # The following code calculates correct decisions and errors for bands.
        if truth == test:
            if (truth == 'young'):
                minore = minore + 1
            elif (truth == 'early adult'):
                g_a = g_a + 1
            elif (truth == 'adult'):
                adulto = adulto + 1
            elif (truth == 'elder'):
                anziano = anziano + 1

        if truth != test:
            if (truth == 'young'):
                e_minore = e_minore + 1
            elif (truth == 'early adulto'):
                e_g_a = e_g_a + 1
            elif (truth == 'adult'):
                e_adulto = e_adulto + 1
            elif (truth == 'elder'):
                e_anziano = e_anziano + 1

    # Creating a vector whose first 4 elements indicate the correct decisions, respectively of the 1, 2, 3 and 4 range.
    # The following 4 elements indicate the wrong decisions, in the order of band 1,2,3,4.
    score = [0 for x in range(8)]
    score[0] = minore
    score[1] = g_a
    score[2] = adulto
    score[3] = anziano
    score[4] = e_minore
    score[5] = e_g_a
    score[6] = e_adulto
    score[7] = e_anziano
    return score


# calcul of score.
def errore_fasce(score, leng):
    ris = 0
    for i in range(0, 4):
        ris += score[i]
    ris = ris / leng
    return ris


# Executing my own functions (above) to have statistical results

mid_error = middle_error(test, Y_vali)
print(mid_error)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')

mid_error_age = middle_error_per_age(test, Y_vali)
print(mid_error_age)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')

big_errors = biggest_errors(mid_error_age, mid_error)
print(big_errors)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')

# Errors per bands

score_fasce = score_per_fasce(test, Y_vali)
print(score_fasce)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')

# Print the score taking the bands into consideration.
err_fasce = errore_fasce(score_fasce, len(test))
print(err_fasce)


# Calculation of the epsilon error.

def epsilon_error(prevision, ground_truth, varianza):
    somma = 0
    leng = 0
    for i in range(0, len(prevision)):
        diff = prevision[i] - ground_truth[i]
        var = float(varianza[i])
        if (var != 0):
            esp = (diff ** 2) / (2 * (var ** 2))
            espneg = -esp
            epsilon = 1 - math.exp(espneg)
            somma = somma + epsilon
            leng = leng + 1
    res = somma / leng
    return res


# saving classifiers
from sklearn.externals import joblib

joblib.dump(svr, 'svr_nome.pkl')

# ---------------------------------------------------GridSearch - -------------------------------------------------------------


tuned_parameters = [{'kernel': ['rbf'], 'C': [500, 1500], 'gamma': [0.1]}]

# init svr
svr = svm.SVR(epsilon=4)
clf = GridSearchCV(svr, tuned_parameters, cv=5, scoring='neg_mean_absolute_error')
clf.fit(X_train_hog, Y_train)

# Printing results obtained
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

# y_true, y_pred = y_test, clf.predict(X_test)
# print(classification_report(y_true, y_pred))
# print()


# ---------------------------------------------------GA - ---------------------------------------------------------------------


# Study GA
# this is the parameter list:
paramgrid = {"kernel": ["rbf"], "C": np.logspace(-9, 9, num=25, base=10), "gamma": np.logspace(-9, 9, num=25, base=10)}

res = EvolutionaryAlgorithmSearchCV(estimator=svm.SVR(epsilon=4),
                                    params=paramgrid,
                                    scoring="accuracy",
                                    population_size=50,
                                    cv=StratifiedKFold(n_splits=2),
                                    generations_number=40,
                                    verbose=1,
                                    n_jobs=6)

res.fit(X_train_hog, Y_train)
res.best_estimator_, res.best_score_, res.best_params_

# -------------------------------------------------TEST - ---------------------------------------------------------------------


# Test example
# Taking hog, age label and variance label

with open("hog_nome.pkl", "rb") as data:
    Hog_prova = pickle.load(data)

with open("y_nome.pkl", "rb") as data:
    y_prova = pickle.load(data)

with open("y_nome.pkl", "rb") as data:
    y_var = pickle.load(data)

# Calc score
mscore = svr.score(Hog_prova, y_prova)
print(mscore)

# calc score and printing prediction
test = svr.predict(Hog_prova)
print(test)
print(y_prova)

# I apply the functions described above, plus the epsilon error in addition

mid_error = middle_error(test, y_prova)
print(mid_error)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')
mid_error_age = middle_error_per_age(test, y_prova)
print(mid_error_age)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')
big_errors = biggest_errors(mid_error_age, mid_error)
print(big_errors)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')
score_fasce = score_per_fasce(test, y_prova)
print(score_fasce)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')
err_fasce = errore_fasce(score_fasce, len(test))
print(err_fasce)
print('\n')
print('------------------------------------------------------------------------------------------------------------')
print('\n')
epsilon = epsilon_error(test, y_prova, y_var)
print(epsilon)
