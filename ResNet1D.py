'''
Author: M73ACat
Date: 2023-03-28
Copyright (c) 2023 by M73ACat, All Rights Reserved. 
Reference: keras.applications.ResNet50V2
'''

from keras.layers import (Activation, Add, BatchNormalization, Conv1D, Dense,
                          GlobalAveragePooling1D, Input)
from keras.models import Model
from keras.optimizers import Nadam


def res_block(x, filters, block_nums, kernel_size=3, stride=1):
    """A residual block.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        block_nums: integer, numbers of block. 
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.

    Returns:
        Output tensor for the residual block.
    """
    for _ in range(block_nums):
        preact = BatchNormalization(
            epsilon=1.001e-5)(x)
        preact = Activation('relu')(preact)

        shortcut = Conv1D(
            4 * filters, 1, strides=stride,padding='same')(preact)

        x = Conv1D(
            filters, 1, strides=1, use_bias=False)(preact)
        x = BatchNormalization(
            epsilon=1.001e-5)(x)
        x = Activation('relu')(x)

        x = Conv1D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            padding='same')(x)
        x = BatchNormalization(
            epsilon=1.001e-5)(x)
        x = Activation('relu')(x)

        x = Conv1D(4 * filters, 1)(x)
        x = Add()([shortcut, x])
    return x
    
if __name__ == '__main__':

    inputs = 2048
    outputs = 8

    x_input  = Input(shape=(inputs,1))
    x = Conv1D(4,3,2,padding='same')(x_input)

    x = res_block(x,filters=4,block_nums=1,stride=2)
    x = res_block(x,filters=4,block_nums=3,stride=1)
    
    x = res_block(x,filters=8,block_nums=1,stride=2)
    x = res_block(x,filters=8,block_nums=3,stride=1)
    
    x = res_block(x,filters=16,block_nums=1,stride=2)
    x = res_block(x,filters=16,block_nums=3,stride=1)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)   
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(outputs,activation='softmax')(x)

    model = Model(inputs=x_input,outputs=x)
    optimizers = Nadam(lr=1e-5)
    model.compile(optimizer = optimizers, loss= 'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
