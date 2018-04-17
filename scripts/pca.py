#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------ IMPORTS ------------------------ #
# import sys for the filename
import sys
# load setup data
import json
# split the dataset for training
from  sklearn.model_selection import train_test_split
# PCA library
from sklearn.decomposition import PCA, KernelPCA
# routine imports
import numpy as np
import matplotlib.pyplot as plt
# for additional details
from datetime import datetime
# to save the data
import pickle
# created module
import utility
# --------------------------------------------------------- #


# -------------------- SETUP & DEFAULTS ------------------ #
# open the setup file
with open('../setup.json', 'r') as f:
     setup = json.load(f)
# load value from : setup.json
reduc_dim = setup["Modules"]
train_size = setup["Training"]/100.
dataset_code = setup["Dataset"]
cell_type = setup["Cell"]
model_type = setup["Model"]
# ------------------------------------------------------- #

# -------------------- LOAD DATA ------------------------ #
print "Loading the data."
#X, _ = utility.load_data(dataset_code, cell_type)
# from datframe to NDarray (numpy)
#X = X.values
##
#change LUSC+LUAD
dataset1, y1 = utility.load_data("LUSC", cell_type, withLabel=True)
# save a dictionary of correspondence from index to gene name
gene_dict = {}
[gene_dict.update({idx: num}) for idx, num in enumerate(list(dataset1))]
dataset2, y2 = utility.load_data("LUAD", cell_type, withLabel=True)
dataset = dataset1.append(dataset2)
y = y1.append(y2)
X, y = dataset.values, y.values
##

# get the number of training samples
input_dim = X.shape[1]
# _normalize_ the data
X = X/X.max()
# divide the data in training and test
X_train, X_test = train_test_split(X, \
    train_size=train_size, test_size=1-train_size)
# ------------------------------------------------------- #

# ----------------------------- TRAINING ----------------------------- #
if (model_type == 'pca'):
    model = PCA(n_components=reduc_dim)
    model.fit(X)
else:
    model = KernelPCA(kernel="rbf", n_components=reduc_dim)
    model.fit(X)
# -------------------------------------------------------------------- #

# --------------------------------- SAVE --------------------------------- #
# create a more meaningful identifier for the backups
#details = ":d"+dataset_code + cell_type
##
#change LUSC+LUAD
details = ":d"+"LU" + cell_type
##
details = details + ":g" + str(datetime.now().day) + ":b" + str(reduc_dim)

# save the model and history data on the filesystem
# it saves the file in the non-git'd up directory
print "Saving the model in /backups as \"%s\""%(model_type+"_"+details)
path = "../backups/"+model_type+"_"+details

# create file name for the classifier
fileName = "%s.skl"%(path)
# open the file for writing
fileObject = open(fileName,'wb')
# this writes the object model to the file named fileObject
pickle.dump(model, fileObject)
# here we close the fileObject
fileObject.close()
# ------------------------------------------------------------------------ #
