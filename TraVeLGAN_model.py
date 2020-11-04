# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import functools
# https://github.com/clementabary/travelgan
nn = tf.keras.layers

class Pad(tf.keras.layers.Layer):

    def __init__(self, paddings, mode='CONSTANT', constant_values=0, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode=self.mode, constant_values=self.constant_values)

class InstanceNormalization(nn.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def downsample(input,filters, kernel_size, weight_decay):

    h = nn.Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=2,
                  padding="same",
                  use_bias=False,
                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h = InstanceNormalization()(h)
    h = nn.LeakyReLU()(h)

    return h

def upsample(input,filters, kernel_size, weight_decay):

    h = nn.Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=2,
                           padding="same",
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h = InstanceNormalization()(h)
    h = nn.ReLU()(h)

    return h

def unet_generator(input_shape=(256, 256, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    e1 = downsample(h, 64, 4, weight_decay) # 128 x 128 x 64
    e2 = downsample(e1, 128, 4, weight_decay)   # 64 x 64 x 128
    e3 = downsample(e2, 256, 4, weight_decay)   # 32 x 32 x 256
    e4 = downsample(e3, 512, 4, weight_decay)   # 16 x 16 x 512
    e5 = downsample(e4, 512, 4, weight_decay)   # 8 x 8 x 512
    e6 = downsample(e5, 512, 4, weight_decay)   # 4 x 4 x 512
    e7 = downsample(e6, 512, 4, weight_decay)   # 2 x 2 x 512
    e8 = downsample(e7, 512, 4, weight_decay)   # 1 x 1 x 512

    d1 = tf.concat([upsample(e8, 512, 4, weight_decay), e7], 3) # 2 x 2 x 1024
    d2 = tf.concat([upsample(d1, 512, 4, weight_decay), e6], 3) # 4 x 4 x 1024
    d3 = tf.concat([upsample(d2, 512, 4, weight_decay), e5], 3) # 8 x 8 x 1024
    d4 = tf.concat([upsample(d3, 512, 4, weight_decay), e4], 3) # 16 x 16 x 1024
    d5 = tf.concat([upsample(d4, 256, 4, weight_decay), e3], 3) # 32 x 32 x 512
    d6 = tf.concat([upsample(d5, 128, 4, weight_decay), e2], 3) # 64 x 64 x 256
    d7 = tf.concat([upsample(d6, 64, 4, weight_decay), e1], 3)  # 128 x 128 x 128

    last = nn.Conv2DTranspose(filters=3,
                              kernel_size=4,
                              strides=2,
                              padding="same")(d7)
    last = tf.nn.tanh(last)

    return tf.keras.Model(inputs=inputs, outputs=last)

def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)
    
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        p = int((3 - 1) / 2)

        h = Pad([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

        h = Pad([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)

        return tf.keras.layers.add([x, h])

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)
    
    # 1
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    h = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

    # 5
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    h = tf.keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.keras.layers.Activation('tanh')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)

def siamese(input_shape=(256, 256, 3), weight_decay=0.00005, num_classes=24):

    h = inputs = tf.keras.Input(input_shape)

    nshape = inputs.get_shape()[1]
    filters = 64
    layer = 1
    minshape = 4
    max_filters = filters * 8

    while nshape > minshape:
        h = nn.Conv2D(filters=filters,
                      kernel_size=4,
                      strides=2,
                      padding="same",
                      use_bias=True if layer == 1 else False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        if layer != 1:
            h = InstanceNormalization()(h)
            h = nn.LeakyReLU()(h)
        else:
            h = nn.LeakyReLU()(h)

        nshape /= 2
        filters=min(2 * filters, max_filters)
        layer += 1
    h = nn.Flatten()(h)

    h = nn.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

class NegativePairSelector():
    def __init__(self):
        pass

    def __call__(self, size):
        return np.asarray(list(combinations(range(size), 2)))