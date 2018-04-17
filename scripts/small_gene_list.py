#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------- IMPORTS -------------- #
#import for cmd line args and filename
import sys, getopt, os
# created module
import utility
# to create one hot encoded v
import numpy as np
# rank results
from scipy.stats import rankdata
# import csv to write in a file
import csv
# load setup data
import json
# use regex to find dataset code
import re
# to save the data
import pickle
# get progress bar to see completion
from tqdm import tqdm, trange
# ------------------------------------- #

# ------------ LOAD PARAMETERS ------------ #
# open the setup file
with open('../modules.json', 'r') as f:
     mod = json.load(f)
# load value from : clf.json
model_name = mod["Model"]
name_format = mod["Format"]
full_scale = mod["Full"]
if full_scale == 0:
    k = mod["K"]
else:
    k = int(re.search("(?<=:st)([0-9]*)", model_name).group(0))
# ----------------------------------------- #

# ------------------------ GET DECODER ---------------------- #
decoder, history = utility.get_net("../backups/"+model_name)
decoder = utility.get_decoder(decoder)
input_dim = decoder.layers[0].get_input_shape_at(0)[1]
# ----------------------------------------------------------- #

# ordered gene grouping
modules = []
# list of all activation level to try from the highest to
# the lowest, anyway the highest include all the lowest
if k == int(re.search("(?<=:st)([0-9]*)", model_name).group(0)):
    activations = [1.]
else:
    activations = np.round(np.linspace(1., .6, 5), decimals=2)

for index in trange(input_dim, desc="Dimensions "):
    # get the index position for the activated genes
    # for all th tried activation values.
    candidates = []

    # -------------------- HOT ENCODED ACTIVATION -------------------- #
    # choose the activation values equi-spaced
    # highest first because we care about the intersection
    for act in tqdm(activations, desc="Activations "):
        # create an array of all zeros
        onehot = np.zeros(input_dim)
        # substitute the activation value
        onehot[index]=act
        onehot = onehot.reshape((-1, input_dim))
        # gather the values from the decoder
        solution = decoder.predict(onehot)
        # get the gene id directly ranked in decreasing order
        candidates.append([x for _,x in sorted(zip(solution[0], \
            range(len(solution[0]))), reverse=True)])
    # ---------------------------------------------------------------- #

    # --------------------- INTERSECTION --------------------- #
    if k == -1:
        # stays always the same you could just take the fst.
        result = candidates[0]
        for candidate in tqdm(candidates[1:], desc="Intersection "):
            result = list(set(result).intersection(candidate))
    else:
        result = candidates[0][:k]
        for candidate in tqdm(candidates[1:][:k], desc="Intersection "):
            result = list(set(result).intersection(candidate))

    # -------------------------------------------------------- #

    # save the result for this dimension
    modules.append(result)

# --------------------------------- SAVE --------------------------------- #
# we open the file for reading i2g information
dataset_code = re.search("(?<=:d)(all|[A-Z]*)", model_name).group(0)
start_dim = re.search("(?<=:st)([0-9]*)", model_name).group(0)
if k == 20000:
    fileObject = open("../backups/%s.i2g"%(dataset_code),'rb')
else:
    fileObject = open("../backups/%s-%s.i2g"%(dataset_code, start_dim),'rb')
# load the object from the file into var b
gene_dict = pickle.load(fileObject)
# here we close the fileObject
fileObject.close()

# save the results in a csv where the fst is the number of dimensions
print "Saving the the gene modules."
if k ==-1:
    details = ":ALL"
else:
    details = ":k%s"%(k)

if name_format == "string":
    with open("../backups/"+model_name+details+"s.csv", "w") as csvfile:
        matrixwriter = csv.writer(csvfile, delimiter='\t')
        for idx, row in enumerate(modules):
            row = [gene_dict[indexes] for indexes in row]
            row.insert(0, idx)
            matrixwriter.writerow(row)
else:
    with open("../backups/"+model_name+details+"n.csv", "w") as csvfile:
        matrixwriter = csv.writer(csvfile, delimiter='\t')
        for idx, row in enumerate(modules):
            row.insert(0, idx)
            matrixwriter.writerow(row)
# ------------------------------------------------------------------------ #
