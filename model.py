import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    print(inputs.get_shape())
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) #64
    # print(conv1.shape)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) #64
    # print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.shape)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1) #128
    # print(conv2.shape)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) #128
    # print(conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(pool2.shape)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) #256
    # print(conv3.shape)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) #256
    # print(conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print(pool3.shape)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) #512
    # print(conv4.shape)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) #512
    # print(conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    # print(drop4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # print(pool4.shape)
    # print('---------bottom reached !!!!--------')

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4) #1024
    # print(conv5.shape)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) #1024
    # print(conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    # print(drop5.shape)
    # print('-----------start upsampling----------')

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5)) #512
    # print(up6.shape)
    merge6 = concatenate([drop4,up6], axis = 3)
    # print(merge6.shape)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6) #512
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) #512

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6)) #256
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7) #256
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) #256

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7)) #128
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8) #128
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8) #128

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) #64
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9) #64
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) #64
    conv9 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) #2
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9) #1
    # print(conv10.shape)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


