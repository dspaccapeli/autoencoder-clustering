# -*- coding: utf-8 -*-
# ------------------------ IMPORTS ------------------------ #
# to save the data
import pickle
# to load the data
import keras.models
# dataframe manipulation
from pandas import *
# to read all the files in a folder
import os
# ------------------------------------------------------- #

#------------------------------------------------------------------------------#
def save_net(model, history, name):
    """Save the model and its training history into the filesystem.
    param:
        model - trained model to be saved
        history - history objects of the training loss progression
        name - string that identifies the model with parameters
    return:
        null"""
    # ------------------ SERIALIZE HISTORY DATA ------------------ #
    # create file name
    fileName = "%s.hist"%(name)
    # open the file for writing
    fileObject = open(fileName,'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(history.history,fileObject)
    # here we close the fileObject
    fileObject.close()
    # ------------------------------------------------------------ #

    # ------------------ SERIALIZE THE MODEL DATA  ------------------ #
    model.save("%s.h5"%(name))
    # --------------------------------------------------------------- #
#------------------------------------------------------------------------------#
def get_net(name):
    """Get the model and the training history from saved data in the same dir.
    param:
        name - string, backup identifier
    return: tuple (2)
        Model,
        history.history"""

    model = keras.models.load_model("%s.h5"%(name))
    # we open the file for reading
    fileObject = open("%s.hist"%(name),'rb')
    # load the object from the file into var b
    history = pickle.load(fileObject)
    # here we close the fileObject
    fileObject.close()

    return model, history
#------------------------------------------------------------------------------#
def load_data(archive_name, cell_type = "both", withLabel=False):
    """Load the dataset specified in the path argument.
    param:
        archive_name - name of the genomic expression matrix
        cell_type - "both", "cancer", "healthy" refined selection
        withLabel - bool, if you want also the column with the cancer label
    return:
        X - dataframe
        y - dataframe"""
    # ------------------------ SELECT THE DATASET ------------------------ #
    if(archive_name=="all"):
        #scan all files in the parent folder
        dir = os.listdir('..')
        dataset = [file for file in dir if file.isupper()]
        # make a list containing all the dataframes
        dfs = [read_table('../'+db+'/genMat', \
            sep= "\s+", header=0, index_col=0).T for db in dataset]
        dataframe = dfs[0]
        # append all the other dataframess
        for df in dfs[1:]:
            dataframe = dataframe.append(df)
        # eliminate non-shared column --> _any_
        dataframe = dataframe.dropna(axis=1, how='any')
    else:
        dataframe = read_table("../"+archive_name+"/genMat", sep= "\s+", \
            header=0, index_col=0).T
    # -------------------------------------------------------------------- #

    # ------------------ FILTER AND RETURN ------------------ #
    dataframe = filter_data(dataframe, cell_type, withLabel, 'cancerous')

    if (withLabel):
        y = dataframe['cancerous']
        X = dataframe.drop('cancerous', axis=1)
        return  X, y

    # return the dataframe as a numpy array
    return dataframe, None
#------------------------------------------------------------------------------#
def get_decoder(autoencoder):
    """Extract the decoder from the model.
    param:
        autoencoder - keras model
    return:
        Model - decoder"""
    net_size = len(autoencoder.layers)/2
    encoded_input = keras.layers.Input(shape=(autoencoder.layers[-net_size].get_input_shape_at(0)[1],))
    decoder = autoencoder.layers[-net_size](encoded_input)
    for index in reversed(range(net_size)[1:]):
        decoder = autoencoder.layers[-index](decoder)
    # create the decoder model
    return keras.models.Model(encoded_input, decoder)
#------------------------------------------------------------------------------#
def get_encoder(autoencoder):
    """"Extract the encoder from the model.
    param:
        autoencoder - keras model
    return:
        Model - encoder"""
    net_size = len(autoencoder.layers)/2 + 1
    net_input = keras.layers.Input(shape=(autoencoder.layers[1].get_input_shape_at(0)[1],))
    encoder = autoencoder.layers[1](net_input)
    for index in range(net_size)[2:]:
        encoder = autoencoder.layers[index](encoder)
    # create the encoder model
    return keras.models.Model(net_input, encoder)
#------------------------------------------------------------------------------#
def filter_data(dataframe, cell_type, withLabel=False, col_name="cancerous"):
    """Filter the dataframe.
    param:
        dataframe - genomic matrix as a dataframe
        cell_type - "both", "cancer", "healthy" refined selection
        withLabel - bool, if you want also the column with the cancer label
        col_name - string, specify new column name
    return:
        Dataframe - dataset"""
    # ------------------------ PRE-PROCESSING ------------------------ #
    # duplicate the index for filtering purposes
    dataframe[col_name] = dataframe.index
    # insert a column where where 1:cancer_cell and 0:healthy_cell
    dataframe[col_name].replace(regex='.*0[1-9]$', value=1, inplace=True)
    dataframe[col_name].replace(regex='.*1[0-9]$', value=0, inplace=True)
    # eliminate control cell types
    dataframe.query('%s in [0, 1]'%(col_name), inplace=True)
    # ------------------------ RETURN OPTIONS ------------------------ #
    # construct return objects
    if cell_type == "both":
        if withLabel:
            # return dataframe w labels
            return dataframe
        else:
            # return dataframe w/o labels
            return dataframe.drop(col_name, axis=1)
    elif cell_type == "cancer":
        if withLabel:
            # return a dataframe with cancerous samples w the label
            return dataframe[dataframe[col_name]==1]
        else:
            # return a dataframe with cancerous samples w/o the label
            return dataframe[dataframe[col_name]==1].drop(col_name, axis=1)
    else:
        if withLabel:
            # return a dataframe with healthy samples w the label
            return dataframe[dataframe[col_name]==0]
    # return a dataframe with healthy samples w/o the label
    return dataframe[dataframe[col_name]==0].drop(col_name, axis=1)
#------------------------------------------------------------------------------#
