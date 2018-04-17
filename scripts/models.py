# ------- IMPORTS ----- #
# import keras
import keras.layers
import keras.models
# import numpy (linspace)
import numpy
# --------------------- #

# class containing a collection of the used models
class Autoencoder:
    def dense(self, input_dim, reduc_dim, activation_function, layers_num):
        '''
            Autoencoder model where every layer is fully connected to the other.
        '''
        # ------------------------- MODEL DEFINITION ------------------------ #
        # creating the input object for the autoencoder
        autoencoder_input = keras.layers.Input(shape=(input_dim,))
        # stacking the encoder layers
        # scale the dimensionality reduction as a linspace
        dim_progression = numpy.linspace(input_dim, \
            reduc_dim, layers_num+1)[1:]
        # you must keep the named reference for the first layer
        autoencoder_part = keras.layers.Dense(int(dim_progression[0]), \
            activation=activation_function)(autoencoder_input)
        for size in dim_progression[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function)(autoencoder_part)
        # stacking the decoder layers
        for size in numpy.linspace(reduc_dim, input_dim, layers_num+1)[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function)(autoencoder_part)
        # this model maps an input to its reconstruction
        autoencoder_model = keras.models.Model(autoencoder_input, \
            autoencoder_part)
        # return the instanciated autoencoder model
        return autoencoder_model
    #--------------------------------------------------------------------------#
    def sparse(self, input_dim, reduc_dim, activation_function, layers_num):
        '''
            Autoencoder model with regularization.
        '''
        # ------------------------- MODEL DEFINITION ------------------------ #
        # creating the input object for the autoencoder
        autoencoder_input = keras.layers.Input(shape=(input_dim,))
        # stacking the encoder layers
        # scale the dimensionality reduction as a linspace
        dim_progression = numpy.linspace(input_dim, \
            reduc_dim, layers_num+1)[1:]
        # you must keep the named reference for the first layer
        autoencoder_part = keras.layers.Dense(int(dim_progression[0]), \
            activation=activation_function, \
            activity_regularizer=keras.regularizers.l1(10e-7))(autoencoder_input)
        for size in dim_progression[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function, \
                activity_regularizer=keras.regularizers.l1(10e-7))(autoencoder_part)
        # stacking the decoder layers
        for size in numpy.linspace(reduc_dim, input_dim, layers_num+1)[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function, \
                activity_regularizer=keras.regularizers.l1(10e-7))(autoencoder_part)
        # this model maps an input to its reconstruction
        autoencoder_model = keras.models.Model(autoencoder_input, \
            autoencoder_part)
        # return the instanciated autoencoder model
        return autoencoder_model
    #--------------------------------------------------------------------------#
    def batch_normalized(self, input_dim, reduc_dim, activation_function, layers_num):
        '''
            Autoencoder model where in-between every layer
            there is a normalization layer.
        '''
        # ------------------------- MODEL DEFINITION ------------------------ #
        # creating the input object for the autoencoder
        autoencoder_inputs = keras.layers.Input(shape=(input_dim,))
        # stacking the encoder layers
        # scale the dimensionality reduction as a linspace
        dim_progression = numpy.linspace(input_dim, reduc_dim, \
            layers_num+1)[1:]
        # you must keep the named reference for the first layer
        autoencoder_part = keras.layers.Dense(int(dim_progression[0]), \
            activation=activation_function)(autoencoder_input)
        autoencoder_part = keras.layers.BatchNormalization()(autoencoder_part)
        for size in dim_progression[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function)(autoencoder_part)
            if (int(size) != reduc_dim):
                autoencoder_part = keras.layers.BatchNormalization()(autoencoder_part)
        # this model maps an input to its reconstruction
        autoencoder_model = keras.models.Model(keras.layers.Input( \
            shape=(input_dim,)), autoencoder_part)
        # return the instanciated autoencoder model
        return autoencoder_model
    #--------------------------------------------------------------------------#
    def sparse_mid(self, input_dim, reduc_dim, activation_function, layers_num):
        '''
            Autoencoder model with regularization.
        '''
        # ------------------------- MODEL DEFINITION ------------------------ #
        # creating the input object for the autoencoder
        autoencoder_input = keras.layers.Input(shape=(input_dim,))
        # stacking the encoder layers
        # scale the dimensionality reduction as a linspace
        dim_progression = numpy.linspace(input_dim, \
            reduc_dim, layers_num+1)[1:]
        # you must keep the named reference for the first layer
        autoencoder_part = keras.layers.Dense(int(dim_progression[0]), \
            activation=activation_function)(autoencoder_input)
        for size in dim_progression[1:-1]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function)(autoencoder_part)
        # MID LAYER
        autoencoder_part = keras.layers.Dense(int(dim_progression[-1]), \
            activation=activation_function, \
            activity_regularizer=keras.regularizers.l1(10e-3))(autoencoder_part)
        # stacking the decoder layers
        for size in numpy.linspace(reduc_dim, input_dim, layers_num+1)[1:]:
            autoencoder_part = keras.layers.Dense(int(size), \
                activation=activation_function)(autoencoder_part)
        # this model maps an input to its reconstruction
        autoencoder_model = keras.models.Model(autoencoder_input, \
            autoencoder_part)
        # return the instanciated autoencoder model
        return autoencoder_model
    #--------------------------------------------------------------------------#
