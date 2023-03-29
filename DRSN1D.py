'''
Author: M73ACat
Date: 2023-03-28
Copyright (c) 2023 by M73ACat, All Rights Reserved. 
Reference: https://github.com/zhao62/Deep-Residual-Shrinkage-Networks
'''
import keras
from keras import backend as K
from keras.layers import (Activation, AveragePooling1D, BatchNormalization,
                          Conv1D, Dense, GlobalAveragePooling1D, Input)
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Nadam
from keras.regularizers import l2


def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(inputs,1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs)
    inputs = K.spatial_2d_padding(inputs,padding=((0,0),(pad_dim,pad_dim)))
    return K.squeeze(inputs,-1)

def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    """A residual_shrinkage_block.

    Arguments:
        incoming: input tensor.
        nb_blocks: integer, numbers of block. 
        out_channels: integer, filters of the conv1d layer.
        downsample: default False, downsample or not.
        downsample_strides: default 2, stride of the first layer.

    Returns:
        Output tensor for the residual block.
    """
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for _ in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, strides=downsample_strides, 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])
        
        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling1D(pool_size=1, strides=2)(identity)
            
        # Zero_padding or Conv1D to match channels
        if in_channels != out_channels:
            """ padding """
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
            """ Conv1D """
            # identity = Conv1D(out_channels,1,strides=1,padding='same')(identity)

        residual = keras.layers.add([residual, identity])
    
    return residual

if __name__ == '__main__':
    
    inputs = 2048
    outputs = 8

    x_input = Input(shape=(inputs,1))
    x = Conv1D(4,3,2,padding='same')(x_input)
    x = residual_shrinkage_block(x, 1, 4, downsample=True)
    x = residual_shrinkage_block(x, 3, 4, downsample=False)

    x = residual_shrinkage_block(x, 1, 8, downsample=True)
    x = residual_shrinkage_block(x, 3, 8, downsample=False)
    
    x = residual_shrinkage_block(x, 1, 16, downsample=True)
    x = residual_shrinkage_block(x, 3, 16, downsample=False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = GlobalAveragePooling1D()(x)

    x = Dense(outputs,activation='softmax')(x)
    
    model = Model(inputs=x_input,outputs=x)
    optimizers = Nadam(lr=1e-5)
    model.compile(optimizer = optimizers, loss= 'categorical_crossentropy',metrics=['accuracy'])
    model.summary()