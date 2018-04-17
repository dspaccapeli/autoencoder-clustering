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
# ------------------------------------- #

# ------------ LOAD PARAMETERS ------------ #
# open the setup file
with open('../enrich.json', 'r') as f:
     enrich = json.load(f)
# load value from : enrich.json
model_name = enrich["Model"]
k = enrich["K"]
batch = enrich["Batch"]
# ----------------------------------------- #

# ------------ LOAD PROGRESS ------------ #
# open the setup file
with open('../enr_prg.json', 'r') as f:
    data = json.load(f)
start_here = data['StartHere']
# --------------------------------------- #


# File name.
csv_path ="../backups/" + model_name + ".csv"

# Reading the stored csv file to list.
lst = []
with open(csv_path, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        lst.append(row[1:])

#dbs = ['GO_Cellular_Component_2015','GO_Biological_Process_2015',\
#    'GO_Molecular_Function_2015','KEGG_2016']

dbs = ['GO_Cellular_Component_2015','GO_Biological_Process_2015',\
    'GO_Molecular_Function_2015']

while True:
    try:
        # Dictionary containing all the enriched dfs.
        save_file = {}
        # Default values for biggest/smallest search values.
        # Reasonably they'll be overwritten.
        smll_p = 1
        bgst_ol = 0
        for idx, module in enumerate(tqdm(lst, desc="Module ")):
            # Skip until you catch up to the progress.
            if idx < start_here:
                continue
            '''
            # Break to not overload the server.
            if idx == (start_here + batch):
                break
            '''

            # Enrich the modules for all (4) the gene lists.
            for db in dbs:
                enr = gseapy.enrichr(gene_list=[x for x in module[:k]], \
                    description='pathway', gene_sets=db, outdir='test')
                df = enr.res2d.iloc[:5,:6]
                # Find the highest overlap.
                for values in enr.res2d['Overlap']:
                    # Split the string.
                    num, den = values.split("/")
                    # Compute and compare the ratio numerically.
                    if (float(num)/float(den) > bgst_ol):
                        bgst_ol = float(num)/float(den)
                        db_ol = db
                        mod_ol = idx
                # Find the smallest p-value.
                for values in enr.res2d['P-value']:
                    if (float(values)<smll_p):
                        smll_p = float(values)
                        db_p = db
                        mod_p = idx
                # Prepare the suffix for appropriate storage.
                if (db == 'GO_Cellular_Component_2015'):
                    suffix = '-cell'
                elif (db == 'GO_Biological_Process_2015'):
                    suffix = '-bio'
                elif (db == 'GO_Molecular_Function_2015'):
                    suffix = '-mol'
                elif (db == 'KEGG_2016'):
                    suffix = '-kegg'
                # Save in the dictionary to later save to disk.
                save_file[str(idx)+suffix] = enr.res2d.iloc[:5,:6]
                # Wait so not to overwhelm the server.
                time.sleep(23)

            if not idx % batch and idx!=start_here or idx==len(lst)-1:
                    # ----------------- SAVE ENRICHMENT ----------------- #
                    fileName = "../backups/%s.%s-%s"%(model_name, \
                    start_here, idx)
                    # open the file for writing
                    fileObject = open(fileName,'wb')
                    # this writes the object a to the
                    # file named 'testfile'
                    pickle.dump(save_file,fileObject)
                    # here we close the fileObject
                    fileObject.close()
                    # ------------------------------------------------- #

                    # ------------ SAVE PROGRESS ------------ #
                    data['StartHere'] = idx
                    with open('../enr_prg.json', 'w') as f:
                        json.dump(data, f)
                    # --------------------------------------- #

                    # --------------------------- SAVE SUMMARY --------------------------- #
                    with open("../backups/%s.enr%s-%s"%(model_name, \
                        start_here, idx), "w") as text_file:
                        sys.stdout = text_file
                        print "The biggest Overlap of " + str(bgst_ol) + " was seen for",
                        print str(db_ol) + " and " + str(mod_ol) + " module."

                        print "The smallest P-Value of " + str(smll_p) + " was seen for",
                        print str(db_p) + " and " + str(mod_p) + " module."
                    # --------------------------------------------------------------------- #

                    start_here = idx
                    smll_p = 1
                    bgst_ol = 0
                    save_file = {}
                    #time.sleep(1020)
                    time.sleep(600)
            if idx==len(lst)-1:
                sys.exit(1)
    except SystemExit as e:
        sys.exit(e)
    except:
        pass
