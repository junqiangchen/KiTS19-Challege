'''
covlution layer，pool layer，initialization。。。。
'''
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, activefunction='sigomd', uniform=True, variable_name=None):
    if activefunction == 'sigomd':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
    elif activefunction == 'relu':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
    elif activefunction == 'tan':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * 4
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * 4
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# Bias initialization
def bias_variable(shape, variable_name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# 2D convolution
def conv2d(x, W, strides=1):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return conv_2d


def normalizationlayer(x, is_train, height=None, width=None, norm_type='None', G=16, esp=1e-5, scope=None):
    with tf.name_scope(scope + norm_type):
        if norm_type == 'None':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train)
        elif norm_type == 'group':
            # tranpose:[bs,h,w,c]to[bs,c,h,w]follwing the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            if H == None and W == None:
                H, W = height, width
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gama and beta
            gama = tf.get_variable(scope + norm_type + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(scope + norm_type + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
            gama = tf.reshape(gama, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])
            output = tf.reshape(x, [-1, C, H, W]) * gama + beta
            # tranpose:[bs,c,h,w]to[bs,h,w,c]follwing the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        return output


# 2D downsampling
def downsampled2d(x, scale_factor, scope=None):
    x_shape = tf.shape(x)
    k = tf.ones([scale_factor, scale_factor, x_shape[-1], x_shape[-1]])
    # note k.shape = [rows, cols, depth_in, depth_output]
    downsampl = tf.nn.conv2d(x, filter=k, strides=[1, scale_factor, scale_factor, 1], padding='SAME', name=scope)
    return downsampl


# 2D upsampling
def upsample2d(x, scale_factor, scope=None):
    ''''
    X shape is [nsample,rows, cols, channel]
    out shape is[nsample,rows*scale_factor, cols*scale_factor, channel]
    '''
    x_shape = tf.shape(x)
    k = tf.ones([scale_factor, scale_factor, x_shape[-1], x_shape[-1]])
    # note k.shape = [rows, cols, depth_in, depth_output]
    output_shape = tf.stack([x_shape[0], x_shape[1] * scale_factor, x_shape[2] * scale_factor, x_shape[-1]])
    upsample = tf.nn.conv2d_transpose(value=x, filter=k, output_shape=output_shape,
                                      strides=[1, scale_factor, scale_factor, 1],
                                      padding='SAME', name=scope)
    return upsample


# 2D deconvolution
def deconv2d(x, W, stride=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] // stride])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


# Resnet add
def resnet_Add(x1, x2):
    """
x1 shape[-1] is small x2 shape[-1]
    """
    if x1.get_shape().as_list()[3] != x2.get_shape().as_list()[3]:
        # Option A:zero-padding
        residual_connection = x2 + tf.pad(x1, [[0, 0], [0, 0], [0, 0],
                                               [0, x2.get_shape().as_list()[3] - x1.get_shape().as_list()[3]]])
    else:
        residual_connection = x2 + x1
        # residual_connection=tf.add(x1,x2)
    return residual_connection


def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    result = merge_img * 255.
    result = np.clip(result, 0, 255).astype('uint8')
    return cv2.imwrite(path, result)
