#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 1st line 'shebang' makes it an exec
# 2nd avoid problem with comments' coding

# ----------------------- IMPORTS ----------------------- #
# import sys for the filename
import sys
# load setup data
import json
# created module
import utility
import models
# split the dataset for training
from  sklearn.model_selection import train_test_split
# used for early stopping durring training
import keras.callbacks
# to use less thread
from keras import backend as K
# check if you can use max thread and additional details
from datetime import datetime
# to save the data
import pickle
# use svc as base classifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
# use optunity for parameters tweaking
import optunity
import optunity.metrics
import sklearn.svm
# to get complete confusion matrix
from pandas_ml import ConfusionMatrix
# to get proper naming & load setup data
import json
import sys
# split the dataset for training
from  sklearn.model_selection import train_test_split
# PCA library
from sklearn.decomposition import PCA, KernelPCA
# routine imports
import numpy as np
# ------------------------------------------------------- #

# ------------------ SETUP & DEFAULTS ---------------- #
# open the setup file
with open('../setup.json', 'r') as f:
     setup = json.load(f)
# load value from : setup.json
reduc_dim = setup["Modules"]
epochs_num = setup["Epochs"]
batch_dim = setup["Batch"]
shuffle_bol = json.loads(setup["Shuffle"].lower())
verb_type = setup["Verbose"]
train_size = setup["Training"]/100.
dataset_code = setup["Dataset"]
model_type = setup["Model"]
min_delta = setup["Delta"]
patience =  setup["Patience"]
optimizer =  setup["Optimizer"]
loss =  setup["Loss"]
activation_function = setup["Activation"]
start_dim = setup["Starting"]
# layers for decoder(/encoder)
# if 2 the overall net will have 4 layers
layers_num = setup["Layer"]
cell_type = setup["Cell"]
# ---------------------------------------------------- #


# ---------------------------- LOAD DATA -------------------------------- #
print "Loading the data."
dataset, y = utility.load_data(dataset_code, "both", withLabel=True)
while start_dim < dataset.shape[1]/2 and start_dim != 20000:
    dataset = dataset.loc[:, dataset.var() > dataset.var().median()]
print "Pre-filtering dimensions " + str(dataset.shape[1])
# reduce intial dimensions
exit_flag = False
while True and start_dim != 20000:
    min_var = dataset.var().min()
    for i in dataset:
        if dataset.shape[1] <= start_dim:
            exit_flag = True
            break
        if dataset[i].var() == min_var:
            del(dataset[i])
    if exit_flag:
        break
print "The starting dimensions are now " + str(dataset.shape[1])
start_dim = dataset.shape[1]
# save a dictionary of correspondence from index to gene name
gene_dict = {}
[gene_dict.update({idx: num}) for idx, num in enumerate(list(dataset))]
############### MATTEUCCI
shift_X = dataset+1
logdf = shift_X.apply(np.log)
logdf[logdf==float('-Inf')] = 0
min_max_scaler = preprocessing.MinMaxScaler()
dataset = min_max_scaler.fit_transform(logdf)
###############
y=y.values
# get the number of training samples
input_dim = dataset.shape[1]
# divide the data in training and test for OVERALL TRAINING
mod_X_train, all_X_test, mod_y_train, all_y_test = train_test_split(dataset, y, \
    train_size=train_size, test_size=1-train_size, stratify=y, random_state=21)
# divide the data in training and test for AUTOENCODER TRAINING
X_train, X_test, y_train, y_test = train_test_split(mod_X_train, mod_y_train, \
    train_size=train_size, test_size=1-train_size, stratify=mod_y_train, random_state=21)
# ----------------------------------------------------------------------- #

# ------------------------- LOAD MODEL ------------------------- #
# load correct instanced method from the name
print "Loading the model \"%s\"."%(model_type)
load_autoencoder = getattr(models.Autoencoder(), model_type)
# store all the correct models
autoencoder = load_autoencoder(input_dim, reduc_dim, \
    activation_function, layers_num)
# -------------------------------------------------------------- #


# --------------------------------- TRAINING --------------------------------- #
# configure the model into a working prototype for training
print "Compiling the Autoencoder."
# og. optimizer adadelta, now sgd as the paper
# og. loss binary_crossentropy, now mean_squared_error
autoencoder.compile(optimizer=optimizer, loss=loss)
# limit thread usage only at a certain hour
# arg was: datetime.now().hour>=21, now don't do it unless really needed
if (False):
    print "Running at night."
    # if you run it a night you have more power and time
    epochs_num = epochs_num*2
    patience = patience*2
else:
    # limit thread usage
    K.set_session(K.tf.Session(config=K.tf.ConfigProto( \
        intra_op_parallelism_threads=18, \
        inter_op_parallelism_threads=18)))
print "Fitting the net."
# train following the specified values
history = autoencoder.fit(mod_X_train, mod_X_train,
                epochs=epochs_num,
                batch_size=batch_dim,
                shuffle=shuffle_bol,
                verbose=verb_type,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', \
                    min_delta=min_delta, patience=patience, \
                    verbose=verb_type, mode='min')],
                validation_data=(X_test, X_test))
# ---------------------------------------------------------------------------- #
print "Computing the PCA."
model_pca = PCA(n_components=reduc_dim)
model_pca.fit(X_train)
print "Computing the kPCA."
model_kpca = KernelPCA(kernel="rbf", n_components=reduc_dim)
model_kpca.fit(X_train)

# ----------------------------------- SAVE ----------------------------------- #
# create a more meaningful identifier for the backups
details = "l" + str(layers_num) + ":d"+dataset_code + cell_type
details = details + ":e" + str(epochs_num) + ":f"+str(shuffle_bol).lower()[0]
details = details + ":l" + loss + ":o" + optimizer + ":a" + activation_function
# string for MATTEUCCI model
details = details + ":g" + str(datetime.now().day) + ":b" + str(reduc_dim) + "matt"

# save the model and history data on the filesystem
# it saves the file in the non-git'd up directory
print "Saving the model in /backups as \"%s\""%(model_type+"_"+details)
utility.save_net(autoencoder, history, "../backups/"+model_type+"_"+details)

# create file name for the classifier
fileName = "../backups/%s.i2g"%(dataset_code)
# open the file for writing
fileObject = open(fileName,'wb')
# this writes the object a to the
# file named 'testfile'
pickle.dump(gene_dict, fileObject)
# here we close the fileObject
fileObject.close()
# ---------------------------------------------------------------------------- #

# # ------------------ SETUP COMPARISON ------------------ #
# # open the setup file
# with open('../classification.json', 'r') as f:
#      clf = json.load(f)
# # load value from : clf.json
# model_type = clf["Model"]
# model_path = "../backups/" + clf["Model_Path"]
# dataset_code = clf["Dataset"]
# cell_type = clf["Cell"]
# train_size = clf["Training"]/100.
# # ------------------------------------------------------ #

# -------------- CALCULATING REDUCED DIMENSIONS -------------- #
model_ae = autoencoder
print "Calculating reduced dimensions."
ae_X = model_ae.predict(mod_X_train)
pca_X = model_pca.transform(mod_X_train)
kpca_X = model_kpca.transform(mod_X_train)
# ------------------------------------------------------------ #

train_size = 0.6
# -------------------------- CROSS VALIDATE AE ------------------------------ #
# divide the data in training and test
print "Split AE"
ae_X_train, ae_X_test, ae_y_train, ae_y_test = train_test_split(ae_X, mod_y_train, \
    train_size=train_size, test_size=1-train_size, stratify=mod_y_train, random_state=21)
#strata_ae = [np.where(np.array(ae_y_train)==0)[0]]
# ----------------------------- AE ---------------------------------------- #
# from here on taken heavily from the library docs
# score function: twice iterated 3-fold cross-validated accuracy
print "Cross validate AE"
@optunity.cross_validated(x=ae_X_train, y=ae_y_train, num_folds=3, num_iter=2)
def ae_svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, \
        gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(ae_svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
ae_optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(ae_X_train, ae_y_train)
# --------------------------- PCA----------------------------------------- #
# from here on taken heavily from the library docs
# score function: twice iterated 3-fold cross-validated accuracy
# -------------------------- CROSS VALIDATE PCA ----------------------------- #
# divide the data in training and test

print "Split PCA"
pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(pca_X, mod_y_train, \
    train_size=train_size, test_size=1-train_size, stratify=mod_y_train, random_state=21)
#strata_pca = [np.where(np.array(pca_y_train)==0)[0]]
print "Cross validate PCA"
@optunity.cross_validated(x=pca_X_train, y=pca_y_train, num_folds=3, num_iter=2)
def pca_svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, \
        gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(pca_svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
pca_optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(pca_X_train, pca_y_train)
# -------------------------  kPCA --------------------------------------- #
# from here on taken heavily from the library docs
# score function: twice iterated 3-fold cross-validated accuracy
# ------------------------- CROSS VALIDATE kPCA ----------------------------- #
print "Split kPCA"
# divide the data in training and test
kpca_X_train, kpca_X_test, kpca_y_train, kpca_y_test = train_test_split(kpca_X, mod_y_train, \
    train_size=train_size, test_size=1-train_size, stratify=mod_y_train, random_state=21)
#strata_kpca = [np.where(np.array(kpca_y_train)==0)[0]]

print "Cross validate kPCA"
@optunity.cross_validated(x=kpca_X_train, y=kpca_y_train, num_folds=3, num_iter=2)
def kpca_svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, \
        gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(kpca_svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
kpca_optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(kpca_X_train, kpca_y_train)
# ------------------------------------------------------------------------ #

test_ae_X = model_ae.predict(all_X_test)
test_pca_X = model_pca.transform(all_X_test)
test_kpca_X = model_kpca.transform(all_X_test)

# ------------- AE TEST ----------------- #
print "Classify on test set AE"
# classify the test set
ae_y_pred = ae_optimal_model.predict(test_ae_X)
# --------------------------------------- #
# ------------------------- SAVE CONFUSION MATRIX ---------------------------- #
# create confusion matrix to compare real results to predicted ones
confusion_matrix = ConfusionMatrix(all_y_test, ae_y_pred)
# save the results with the models in backups
try:
    fileObject = open("../backups/"+"AE.clf", 'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(confusion_matrix, fileObject)
    # here we close the fileObject
    fileObject.close()
except:
    pass
'''
with open("../backups/"+ "AE.clf", "w") as text_file:
    try:
        text_file.write(confusion_matrix.print_stats())
    except:
        pass
'''
# ---------------------------------------------------------------------------- #

# ------------- PCA TEST ----------------- #
print "Classify on test set PCA"
# classify the test set
pca_y_pred = pca_optimal_model.predict(test_pca_X)
# ---------------------------------------- #
# ------------------------- SAVE CONFUSION MATRIX ---------------------------- #
# create confusion matrix to compare real results to predicted ones
confusion_matrix = ConfusionMatrix(all_y_test, pca_y_pred)
# save the results with the models in backups
try:
    fileObject = open("../backups/"+"PCA.clf", 'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(confusion_matrix, fileObject)
    # here we close the fileObject
    fileObject.close()
except:
    pass
'''
with open("../backups/"+"PCA.clf", "w") as text_file:
    try:
        text_file.write(confusion_matrix.print_stats())
    except:
        pass
'''
# ---------------------------------------------------------------------------- #

# ------------- kPCA TEST ----------------- #
print "Classify on test set kPCA"
# classify the test set
kpca_y_pred = kpca_optimal_model.predict(test_kpca_X)
# ---------------------------------------- #
# ------------------------- SAVE CONFUSION MATRIX ---------------------------- #
# create confusion matrix to compare real results to predicted ones
confusion_matrix = ConfusionMatrix(all_y_test, kpca_y_pred)
# save the results with the models in backups
try:
    fileObject = open("../backups/"+"KPCA.clf", 'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(confusion_matrix, fileObject)
    # here we close the fileObject
    fileObject.close()
except:
    pass
'''
with open("../backups/"+"KPCA.clf", "w") as text_file:
    try:
        text_file.write(confusion_matrix.print_stats())
    except:
        pass
'''
# ---------------------------------------------------------------------------- #
