import keras.backend as K
from keras.optimizers import *
from keras.models import Model
from keras.layers import concatenate, add, Activation, BatchNormalization, core, Dropout, Input, Dense, merge, Conv2D, Conv3D, Conv1D, UpSampling1D, UpSampling2D, UpSampling3D, Flatten, Reshape
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, MaxPooling1D, MaxPooling2D
from keras.layers.core import Lambda
import numpy as np

#
# def spatial_temporal_loss(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def temporal_loss(y_true, y_pred):
    y_pred = tf.transpose(y_pred, [0,2,1])
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse


def metric_cor(y_true, y_pred):
    n = K.sum(K.ones_like(y_true))
    sum_x = K.sum(y_true)
    sum_y = K.sum(y_pred)
    sum_x_sq = K.sum(K.square(y_true))
    sum_y_sq = K.sum(K.square(y_pred))
    psum = K.sum(y_true * y_pred)
    num = psum - (sum_x * sum_y / n)
    den = K.sqrt((sum_x_sq - K.square(sum_x) / n) *(sum_y_sq - K.square(sum_y) / n))

    return tf.cond(tf.is_nan(num/den), lambda : 0., lambda: num/den)

def get_mean(x, batch_size, n_ch):
    return tf.reduce_mean(tf.reshape(x, [batch_size, -1, n_ch]), axis=1, keep_dims=True)

def weighting(input, time_series):
    time_series = tf.expand_dims(time_series, axis=1)
    time_series = tf.expand_dims(time_series, axis=1)
    return tf.multiply(input, time_series)




def time_conv(x, batch_size):
    bs = batch_size
    vol_x = x._keras_shape[1]
    vol_y = x._keras_shape[2]
    vol_z = x._keras_shape[3]
    n_ch = x._keras_shape[4]

    x = tf.expand_dims(tf.reshape(x, [bs*vol_x*vol_y*vol_z, n_ch]),axis=-1)
    x = Conv1D(4, 20, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(8, 12, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, 12, activation='relu', padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(8, 12, activation='relu', padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(1, 20, activation='relu', padding='same')(x)

    return tf.reshape(x, [bs, vol_x, vol_y, vol_z, n_ch])

def get_net_time_weighted(batch_size, patch_z, patch_height, patch_width, n_ch):
    inputs = Input((patch_z, patch_height, patch_width, n_ch))
    # temporal branch
    t_pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='t_pool1')(inputs)

    t_conv1 = Lambda(lambda x: time_conv(x, batch_size))(t_pool1)

    time_series = Lambda(lambda x: get_mean(x, batch_size, x._keras_shape[-1]))(t_conv1)

    # spatial branch
    weighted_channels = Lambda(lambda (input, ts): weighting(input, ts))([inputs, time_series])
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(weighted_channels)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = BatchNormalization(axis=-1)(conv4)

    # upper
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(pool4), conv3], axis=-1)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)

    conv10 = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(conv9)


    # model
    model = Model(inputs=[inputs], outputs=[conv10, time_series])

    # model.compile(optimizer='Adadelta', loss='mean_absolute_error',loss_weights=[1,0.5,0.3,0.1])
    model.compile(optimizer='Adadelta', loss=['mse', temporal_loss])

    return model


def time_conv_with_spatial(x, DMN, batch_size):
    bs = batch_size
    vol_x = x._keras_shape[1]
    vol_y = x._keras_shape[2]
    vol_z = x._keras_shape[3]
    n_ch = x._keras_shape[4]

    x = tf.reshape(x, [bs,vol_x*vol_y*vol_z, n_ch])
    DMN = tf.reshape(DMN, [bs, -1, 1]) #[bs, x*y*z, 1]
    DMN = tf.maximum(DMN-0.1,0)

    x = tf.multiply(x, DMN) #[bs,x*y*z, n_ch]
    x = tf.reduce_mean(x, axis=1) #[bs,n_ch]
    x = tf.expand_dims(x, axis=-1)  # [bs,n_ch,1]

    x = tf.expand_dims(x, axis=-1)  # [bs,n_ch,1,1]
    return x


def get_net_only_spatial(batch_size, patch_z, patch_height, patch_width, n_ch):
    inputs = Input((patch_z, patch_height, patch_width, n_ch))


    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = BatchNormalization(axis=-1)(conv1)
    # pool1 = conv1
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = BatchNormalization(axis=-1)(conv2)
    # pool2 = conv2
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = BatchNormalization(axis=-1)(conv3)
    # pool3 = conv3
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(pool3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = BatchNormalization(axis=-1)(conv4)
    # pool4 = conv4

    # upper
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(pool4), conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)

    conv10 = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(conv9)

    # model
    model = Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer='Adadelta', loss='mean_absolute_error',loss_weights=[1,0.5,0.3,0.1])
    opt = SGD(lr=0.01)
    model.compile(optimizer='Adadelta', loss='mse')

    return model



def get_net_border(batch_size, patch_z, patch_height, patch_width, n_ch, model_pretrained, spatial_trainable):
    model_only_spatial = get_net_only_spatial(batch_size, patch_z, patch_height, patch_width, n_ch, trainable=spatial_trainable)
    model_only_spatial.load_weights(model_pretrained)



    # temporal branch for fine tuning
    time_series = Lambda(lambda x: time_conv_with_spatial(x,model_only_spatial.output, batch_size))(model_only_spatial.input)

    #
    time_series = Lambda(lambda  x: tf.pad(x,tf.constant([[0,0],[20,20],[0,0],[0,0]]), "SYMMETRIC"))(time_series)
    x = Conv2D(8, (5, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(time_series)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (3, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D(size=(2, 1))(x)
    x = Conv2D(16, (3, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization(axis=-1)(x)
    time_series = UpSampling2D(size=(2, 1))(x)
    time_series = Conv2D(1, (5, 1), activation='linear', padding='same', kernel_regularizer=l2(0.001))(time_series)
    time_series = Lambda(lambda x: tf.squeeze(x, axis=-1))(time_series)

    time_series = Lambda(lambda x: x[:,20:-20,:])(time_series)

    # model
    model = Model(inputs=[model_only_spatial.input], outputs=[model_only_spatial.output, time_series])

    # model.compile(optimizer='Adadelta', loss='mean_absolute_error',loss_weights=[1,0.5,0.3,0.1])
    model.compile(optimizer='Adadelta', loss=['mse', metric_cor], loss_weights=[1., -0.1])

    return model