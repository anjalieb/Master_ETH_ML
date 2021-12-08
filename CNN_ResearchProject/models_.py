###########################################################################################
# Libraries
###########################################################################################
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential, Model
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, Activation, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu


###########################################################################################
# Models
###########################################################################################


# 3D Model
def build_CNNflexi_Modular(input):
    """Simple CNN with n Conv layers and either globalAveragePooling or fully connected layers for classification"""

    params_2D    = input['params']
    sample_shape = params_2D['sample_shape']
    inputA       = Input(shape=sample_shape[1:])
    clf          = params_2D['clf']

    # Model head
    for i_layers in range(0,params_2D['n_layers']):
        if i_layers == 0:
            x = conv2D_block(params_2D,inputA)
        else:
            x = conv2D_block(params_2D, x)
            

    # Classifiers
    # if clf == 'dense':
    #     x = Flatten()(x)
    #     x = Dense(256, activation='relu', kernel_initializer='he_uniform',activity_regularizer=l1(params_3D['l1_den']))(x)
    #     x = Dense(128, activation='relu', kernel_initializer='he_uniform',activity_regularizer=l1(params_3D['l1_den']))(x)
    #     x = Dense(64, activation='relu', kernel_initializer='he_uniform',activity_regularizer=l1(params_3D['l1_den']))(x)
    if clf == 'gap':
        x = GlobalAveragePooling2D()(x)
    else:
        raise('Invalid classifier choice!')


    model = Model(inputs=inputA, outputs=x)
    return model

def conv2D_block(params_2D,input):

    if params_2D['BNloc'] == 0:
        x = Conv2D(
            params_2D['n_filters_1'],
            kernel_size=(params_2D['kernel_size'], params_2D['kernel_size']),
            activation=None,
            kernel_initializer='he_uniform',
            input_shape=params_2D['sample_shape'],
            padding='SAME',
            data_format='channels_last',
            kernel_regularizer=l2(params_2D['l2']),
            bias_regularizer=l2(params_2D['l2'])
        )(input)

        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
        x = Activation(relu, activity_regularizer=l1(params_2D['l1_act']))(x)

    else:
        x = Conv2D(
            params_2D['n_filters_1'],
            kernel_size=(params_2D['kernel_size'], params_2D['kernel_size']),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='SAME',
            kernel_regularizer=l2(params_2D['l2']),
            bias_regularizer=l2(params_2D['l2']),
            activity_regularizer=l1(params_2D['l1_act'])
        )(input)

    if params_2D['BNloc'] == 1:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if params_2D['BNloc'] == 2:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)

    if params_2D['useDO']:
        x = Dropout(0.25)(x)

    return x



 # Dense models
def denseOutput(l1_var,shape):

     n_classes = 2
     sample_shape = (shape)
     input = Input(shape=sample_shape)
     x = Dense(n_classes, activation='sigmoid', activity_regularizer=l1(l1_var))(input)
     model = Model(inputs=input, outputs=x)

     return model


# Combined model 2D + TDA
def CNN_Model(input2D,inputTail, choice_model2D):

    #########################################
    #     Top arms of the combined model    #
    #########################################

    # 2D patch model
    if choice_model2D == 'CNNflexi':
         model2D = build_CNNflexi_Modular(input2D)
    else:
        raise('3D model not supported!')


    ###############################
    #           Merging           #
    ###############################
    # Combined input
    combinedInput = model2D.output



    # Tail
    #if inputTail['clf'] == 'clear':
    x = Dense(inputTail['n_classes'], activation='sigmoid')(combinedInput)
    # elif inputTail['clf'] == 'dense':
    #     x = Dense(64, activation='relu', kernel_initializer='he_uniform', activity_regularizer= l1(inputTail['l1']))(combinedInput)
    #     x = Dense(32, activation='relu', kernel_initializer='he_uniform', activity_regularizer= l1(inputTail['l1']))(x)
    #     x = Dense(inputTail['n_classes'], activation='sigmoid')(x)
    # elif inputTail['clf'] == 'gap':
    #     x = Dense(inputTail['n_classes'], activation='sigmoid')(x)
    

    # Combined output
    model = Model(inputs=model2D.inputs, outputs=x)


    return model









