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
from sklearn import preprocessing
# used for early stopping durring training
import keras.callbacks
# to use less thread
from keras import backend as K
# check if you can use max thread and additional details
from datetime import datetime
# to save the data
import pickle
# get progress bar to see completion
from tqdm import tqdm, trange
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
# layers for decoder(/encoder)
# if 2 the overall net will have 4 layers
layers_num = setup["Layer"]
cell_type = setup["Cell"]
start_dim = setup["Starting"]
# ---------------------------------------------------- #

# deep autoencoders for gene clustering
# ---------------------------- LOAD DATA -------------------------------- #
print "Loading the data."
dataset, _ = utility.load_data(dataset_code, cell_type, withLabel=True)
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
# save a dictionary of correspondence from index to gene name
gene_dict = {}
[gene_dict.update({idx: num}) for idx, num in enumerate(list(dataset))]
'''
# from dataframe to NDarray (numpy)
dataset = dataset.values
'''
# get the number of training samples
input_dim = dataset.shape[1]
'''
# _normalize_ the data
dataset = dataset/dataset.max()
# divide the data in training and test
'''
min_max_scaler = preprocessing.MinMaxScaler()
dataset = min_max_scaler.fit_transform(dataset)
X_train, X_test = train_test_split(dataset, \
    train_size=train_size, test_size=1-train_size)
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
print "Compiling the model."
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
history = autoencoder.fit(X_train, X_train,
                epochs=epochs_num,
                batch_size=batch_dim,
                shuffle=shuffle_bol,
                verbose=verb_type,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', \
                    min_delta=min_delta, patience=patience, \
                    verbose=verb_type, mode='min')],
                validation_data=(X_test, X_test))
# ---------------------------------------------------------------------------- #


# ----------------------------------- SAVE ----------------------------------- #
# create a more meaningful identifier for the backups
details = "l" + str(layers_num) + ":d"+dataset_code + cell_type
details = details + ":e" + str(epochs_num) + ":f"+str(shuffle_bol).lower()[0]
details = details + ":l" + loss + ":o" + optimizer + ":a" + activation_function
details = details + ":o" + str(datetime.now().hour) + "m" + str(datetime.now().minute) + ":b" + str(reduc_dim) + ":st" + str(start_dim)

# save the model and history data on the filesystem
# it saves the file in the non-git'd up directory
print "Saving the model in /backups as \"%s\""%(model_type+"_"+details)
utility.save_net(autoencoder, history, "../backups/"+model_type+"_"+details)

# create file name for the classifier
fileName = "../backups/%s-%s.i2g"%(dataset_code, start_dim)
# open the file for writing
fileObject = open(fileName,'wb')
# this writes the object a to the
# file named 'testfile'
pickle.dump(gene_dict, fileObject)
# here we close the fileObject
fileObject.close()
# ---------------------------------------------------------------------------- #
