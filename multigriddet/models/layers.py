#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for YOLOv3 models building
"""
from functools import wraps, reduce

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D, Activation
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# Import from the new structure - we'll define these locally
# from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization

# Define the missing functions locally
def YoloConv2D(*args, **kwargs):
    """YoloConv2D wrapper."""
    return Conv2D(*args, **kwargs)

def YoloDepthwiseConv2D(*args, **kwargs):
    """YoloDepthwiseConv2D wrapper."""
    return DepthwiseConv2D(*args, **kwargs)

def CustomBatchNormalization(*args, **kwargs):
    """CustomBatchNormalization wrapper."""
    return BatchNormalization(*args, **kwargs)

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)

@wraps(YoloDepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloDepthwiseConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloDepthwiseConv2D(*args, **darknet_conv_kwargs)

def Darknet_Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1))


def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))

def mish(x):
    return x * K.tanh(K.softplus(x))

def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and Mish."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        Activation(mish))


def Spp_Conv2D_BN_Leaky(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))([y3, y2, y1, x])
    return y



def make_last_layers(x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            # DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            # DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y

def make_yolo_head(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    return x

def make_yolo_spp_head(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    return x


def make_spp_last_layers(x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y

def make_depthwise_separable_last_layers(x, num_filters, out_filters, block_id_str=None, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_2'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=predict_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y

def make_spp_depthwise_separable_last_layers(x, num_filters, out_filters, block_id_str=None, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_2'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=predict_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y


def denseyolo2_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes, use_spp=False):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    pred_filter = 8 * (num_anchors_per_head[0] + num_classes + 5)

    if use_spp:
        x, y1 = make_spp_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, predict_id='1')
    else:
        x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, pred_filter, predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    pred_filter = 4 * (num_anchors_per_head[0] + num_classes + 5)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors_per_head[1] + num_classes + 5, pred_filter, predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    pred_filter = 2 * (num_anchors_per_head[0] + num_classes + 5)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors_per_head[2] + num_classes + 5, pred_filter, predict_id='3')
    return y1, y2, y3


def yolo4_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head (19x19 for 608 input)
    x1 = make_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (38x38 for 608 input)
    x2 = make_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    #x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, f3_channel_num//2)
    y3 = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num, (3,3)),
            DarknetConv2D(num_anchors_per_head[2] + num_classes + 5, (1,1), name='predict_conv_3'))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    #x2, y2 = make_last_layers(x2, 256, num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors_per_head[1] + num_classes + 5, (1,1), name='predict_conv_2'))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors_per_head[0] + num_classes + 5, (1,1), name='predict_conv_1'))(x1)

    return y1, y2, y3


def denseyolo3_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes, use_spp=False):
    f1, f2, f3, f4, f5 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num, f4_channel_num, f5_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    pred_filter = 32 * (num_anchors_per_head[0] + num_classes + 5)

    if use_spp:
        x, y1 = make_spp_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, predict_id='1')
    else:
        x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, predict_filters=None, predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    pred_filter = 16 * (num_anchors_per_head[1] + num_classes + 5)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors_per_head[1] + num_classes + 5, predict_filters=None, predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    pred_filter = 8 * (num_anchors_per_head[2] + num_classes + 5)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors_per_head[2] + num_classes + 5, predict_filters=None, predict_id='3')

    print("x.shape:1  ",x.shape)
    #upsample fpn merge for feature map 3 & 4
    x = compose(
            DarknetConv2D_BN_Leaky(f4_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    f4 = compose(DarknetConv2D_BN_Leaky(f4_channel_num, (1,1)),
                 UpSampling2D(2))(f4)
    x = Concatenate()([x, f4])
    
    #feature map 3 head & output (52x52 for 416 input)
    pred_filter = 4 * (num_anchors_per_head[3] + num_classes + 5)
    x, y4 = make_last_layers(x, f4_channel_num//2, num_anchors_per_head[3] + num_classes + 5, predict_filters=None, predict_id='4')
    print("f4.shape:2  ",f4.shape)
    #upsample fpn merge for feature map 3 & 4
    x = compose(
            DarknetConv2D_BN_Leaky(f5_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    f5 = compose(DarknetConv2D_BN_Leaky(f5_channel_num, (1,1)),
                 UpSampling2D(4))(f5)
    x = Concatenate()([x, f5])
    
    #feature map 3 head & output (52x52 for 416 input)
    pred_filter = 2 * (num_anchors_per_head[4] + num_classes + 5)
    x, y5 = make_last_layers(x, f5_channel_num//2, num_anchors_per_head[4] + num_classes + 5, predict_filters=None, predict_id='5')
    print("x.shape:3  ",x.shape)
    return y1, y2, y3, y4, y5

def denseyolo2lite_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes, use_spp=False):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    if use_spp:
        x, y1 = make_spp_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + (num_classes + 5), block_id_str='pred_1', predict_id='1')
    else:
        x, y1 = make_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + (num_classes + 5), block_id_str='pred_1', predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_depthwise_separable_last_layers(x, f2_channel_num//2, num_anchors_per_head[1] + (num_classes + 5), block_id_str='pred_2', predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_depthwise_separable_last_layers(x, f3_channel_num//2, num_anchors_per_head[2] + (num_classes + 5), block_id_str='pred_3', predict_id='3')

    return y1, y2, y3


def tiny_denseyolo2_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors_per_head[0]+(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors_per_head[1]+(num_classes+5), (1,1), name='predict_conv_2'))([x2, f2])

    return y1, y2


def tiny_denseyolo2lite_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='pred_1'),
            DarknetConv2D(num_anchors_per_head+(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='pred_2'),
            DarknetConv2D(num_anchors_per_head+(num_classes+5), (1,1), name='predict_conv_2'))([x2, f2])

    return y1, y2

