#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------- IMPORTS -------------- #
# Default imports.
import pandas as pd
import csv
import operator
import json
# import sys for redirecting output
import sys
# import enrichment library
import gseapy
import time
# progress bar
from tqdm import trange, tqdm
# save the Dictionary
import pickle
# take upto next 10
import math
# read all the file in a directory
import os
# ------------------------------------- #

# ------------------ SETUP & DEFAULTS ---------------- #
# open the setup file
with open('../lookup.json', 'r') as f:
     setup = json.load(f)
# load value from : setup.json
query = setup["Gene"]
database = setup["Cancer"]
# ---------------------------------------------------- #

if database == "BREAST":
    code = "BRCA"
elif database == "LUNG":
    code = "LU"
elif database == "KIDNEY":
    code == "KIRC"
elif database == "HEADNECK":
    code == "HNSC"
elif database == "THYROID":
    code == "THCA"
elif database == "PROSTATE":
    code == "PRAD"

direct = os.listdir('../backups/fourLayer/')
model_name = [ x for x in direct if code in x and ".csv" in x and "1000" in x and "l4" in x][0]
csv_path ="../backups/fourLayer/" + model_name

pandalst = pd.read_csv(csv_path, delimiter='\t', header=None)
pandalst = pandalst.drop(pandalst.columns[[0]], axis=1)

module_result = -1
query_index = 1000000
for number, module in pandalst.iterrows():
    for idx, gene in module.iteritems():
        if gene == query:
            module_result = number
            query_index = idx

if module_result == -1:
    print "The gene couldn't be found."
else:
    print "The " +str(module_result)+ "-th module contains the gene."
    print "It was found in its first " +str(query_index)+ " most relevant genes."
    print "The module rounded to the nearest multiple of 10 is the following: "
    print pandalst.as_matrix()[module_result][:int(10*math.ceil(query_index/10.))]
