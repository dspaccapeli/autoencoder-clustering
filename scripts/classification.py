#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------ IMPORTS ------------------------ #
# created module
import utility
# split the dataset for eval
from  sklearn.model_selection import train_test_split
# use svc as base classifier
from sklearn import svm
# use pickle to retrieve files
import pickle
# use optunity for parameters tweaking
import optunity
import optunity.metrics
import sklearn.svm
# to get complete confusion matrix
from pandas_ml import ConfusionMatrix
# to get proper naming & load setup data
import json
import sys
# ------------------------------------------------------- #


# ------------------ SETUP COMPARISON ------------------ #
# open the setup file
with open('../classification.json', 'r') as f:
     clf = json.load(f)
# load value from : clf.json
model_type = clf["Model"]
model_path = "../backups/" + clf["Model_Path"]
dataset_code = clf["Dataset"]
cell_type = clf["Cell"]
train_size = clf["Training"]/100.
# ------------------------------------------------------ #

# ------------ GET SPECIFIED MODEL ------------ #
if (model_type == "ae"):
    # Autoencoder
    model, _ = utility.get_net(model_path[:-3])
    model = utility.get_encoder(model)
elif (model_type == "pca"):
    # PCA
    # open the file for reading
    fileObject = open(model_path, 'rb')
    model = pickle.load(fileObject)
    # here we close the fileObject
    fileObject.close()
else:
    # kPCA
    # open the file for reading
    fileObject = open(model_path, 'rb')
    model = pickle.load(fileObject)
    # here we close the fileObject
    fileObject.close()
# --------------------------------------------- #

# ------------------------- LOAD DATA ----------------------------- #
print "Loading the data."
X, y = utility.load_data(dataset_code, cell_type, withLabel=True)

##
#change LUSC+LUAD
#dataset1, y1 = utility.load_data("LUSC", cell_type, withLabel=True)
# save a dictionary of correspondence from index to gene name
#gene_dict = {}
#[gene_dict.update({idx: num}) for idx, num in enumerate(list(dataset1))]
#dataset2, y2 = utility.load_data("LUAD", cell_type, withLabel=True)
#dataset = dataset1.append(dataset2)
#y = y1.append(y2)
#X, y = dataset.values, y.values
##

# from dataframe to NDarray (numpy)
X, y = X.values, y.values
# get the number of training samples
input_dim = X.shape[1]
# _normalize_ the data
X = X/X.max()
# ----------------------------------------------------------------- #

# --------------------- REDUCE DIMENSIONS ------------------------- #
if (str(type(model)) == "<class 'keras.engine.training.Model'>"):
    X_new = model.predict(X)
else:
    X_new = model.transform(X)
# ----------------------------------------------------------------- #


# -------------------------- CROSS VALIDATE ------------------------------ #
# divide the data in training and test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, \
    train_size=train_size, test_size=1-train_size, random_state=21)


# from here on taken heavily from the library docs
# score function: twice iterated 10-fold cross-validated accuracy

@optunity.cross_validated(x=X_train, y=y_train, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, \
        gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], \
    logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], \
    gamma=10 ** hps['logGamma']).fit(X_train, y_train)
# ------------------------------------------------------------------------ #

# ------------- TEST ----------------- #
# classify the test set
y_pred = optimal_model.predict(X_test)
# ------------------------------------ #


# ------------------------- SAVE CONFUSION MATRIX ---------------------------- #
# create confusion matrix to compare real results to predicted ones
confusion_matrix = ConfusionMatrix(y_test, y_pred)
# save the results with the models in backups
with open("../backups/"+clf["Model_Path"][:-3]+"ts:"+str(int(train_size*100))+".clf", "w") as text_file:
    sys.stdout = text_file
    confusion_matrix.print_stats()
# ---------------------------------------------------------------------------- #
