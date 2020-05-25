"""Deep Transfer Convolutional Neural Network (DTCNN)

Created on:  2019/12/31 22:01

@File: models.py
@Author：Xufeng Huang (xufenghuang1228@gmail.com & xfhuang@umich.edu)
@Copy Right: Copyright © 2019-2020 HUST. All Rights Reserved.
@Requirement: Python-3.7.4, TensorFlow-1.4, Kears-2.2.4

"""
import warnings
import keras.backend as K
from keras.layers import Input, Lambda, GlobalAvgPool2D, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from libs.keras_efficientnets.efficientnet import EfficientNetB5
warnings.filterwarnings("ignore")


# Resnet50
# K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition,
# in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Las Vegas, NV, USA, 2016: pp. 770–778.
# https://doi.org/10.1109/CVPR.2016.90.
def resnet50_model(input_shape=(None, None), num_classes=2):
    input_gray = Input(shape=input_shape)
    input_fakeRgb = Lambda(
        lambda x: K.repeat_elements(
            K.expand_dims(
                x, 3), 3, 3))(input_gray)

    base_model = ResNet50(include_top=False, input_tensor=input_fakeRgb)
    output = GlobalAvgPool2D()(base_model.output)
    predict = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predict)
    return model


# Xception
# F. Chollet, Xception: Deep learning with depthwise separable convolutions,
# in: Proceedings of the IEEE Conference on Computer Vision and Pattern
# Recognition, 2017: pp. 1251–1258.
def xception_model(input_shape=(None, None), num_classes=2):
    input_gray = Input(shape=input_shape)
    input_fakeRgb = Lambda(
        lambda x: K.repeat_elements(
            K.expand_dims(
                x, 3), 3, 3))(input_gray)

    base_model = Xception(include_top=False, input_tensor=input_fakeRgb)
    output = GlobalAvgPool2D()(base_model.output)
    predict = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predict)
    return model


# NASNetMobile
# [1]B. Zoph, V. Vasudevan, J. Shlens, Q.V. Le, Learning transferable architectures for scalable image recognition,
# in: Proceedings of the IEEE Conference on Computer Vision and Pattern
# Recognition, 2018: pp. 8697–8710.
def nasnetmobile_model(input_shape=(None, None), num_classes=2):
    input_gray = Input(shape=input_shape)
    input_fakeRgb = Lambda(
        lambda x: K.repeat_elements(
            K.expand_dims(
                x, 3), 3, 3))(input_gray)

    base_model = NASNetMobile(include_top=False, input_tensor=input_fakeRgb)
    output = GlobalAvgPool2D()(base_model.output)
    predict = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predict)
    return model


# MobileNetV2
# M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks,
# in: Proceedings of the IEEE Conference on Computer Vision and Pattern
# Recognition, 2018: pp. 4510–4520.
def mobilenetv2_model(input_shape=(None, None), num_classes=2):
    input_gray = Input(shape=input_shape)
    input_fakeRgb = Lambda(
        lambda x: K.repeat_elements(
            K.expand_dims(
                x, 3), 3, 3))(input_gray)

    base_model = MobileNetV2(include_top=False, input_tensor=input_fakeRgb)
    output = GlobalAvgPool2D()(base_model.output)
    predict = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predict)
    return model


# EfficientNet-B5
# M. Tan, Q.V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,
# in: International Conference on Machine Learning, 2019.
def efficientnetb5_model(input_shape=(None, None), num_classes=2):
    input_gray = Input(shape=input_shape)
    input_fakeRgb = Lambda(
        lambda x: K.repeat_elements(
            K.expand_dims(
                x, 3), 3, 3))(input_gray)

    base_model = EfficientNetB5(include_top=False, input_tensor=input_fakeRgb)
    output = GlobalAvgPool2D()(base_model.output)
    predict = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predict)
    return model


#Test DTCNN functions
#if __name__ == '__main__':
    #model = resnet50_model((64, 64), 2)
    #model = xception_model((64, 64), 2)
    #model = nasnetmobile_model((64, 64), 2)
    #model = mobilenetv2_model((64, 64), 2)
    #model = efficientnetb5_model((64, 64), 2)
    #model.summary()