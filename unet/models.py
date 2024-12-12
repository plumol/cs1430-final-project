import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from keras.models import Model

def double_conv_layer(x, filters):
    x = Conv2D(filters, kernel_size=3, padding="same", strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=3, padding="same", strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_module(inputs, filters):
    conv_output = double_conv_layer(inputs, filters)
    pooled_output = MaxPool2D(pool_size=(2, 2))(conv_output)
    return conv_output, pooled_output

def decoder_module(inputs, skip_features, filters):
    upsampled = Conv2DTranspose(filters, kernel_size=3, strides=(2, 2), padding="same")(inputs)
    merged = Concatenate()([upsampled, skip_features])
    output = double_conv_layer(merged, filters)
    return output

def unet_model(input_shape):
    inputs = Input(shape=input_shape)

    # encoding
    enc1, pool1 = encoder_module(inputs, 64)
    enc2, pool2 = encoder_module(pool1, 128)
    enc3, pool3 = encoder_module(pool2, 256)
    enc4, pool4 = encoder_module(pool3, 512)

    #Bridge
    bottleneck = double_conv_layer(pool4, 1024)

    # decoding
    dec4 = decoder_module(bottleneck, enc4, 512)
    dec3 = decoder_module(dec4, enc3, 256)
    dec2 = decoder_module(dec3, enc2, 128)
    dec1 = decoder_module(dec2, enc1, 64)

    # output
    outputs = Conv2D(1, kernel_size=1, activation="sigmoid", padding="same")(dec1)

    return Model(inputs, outputs, name="Custom_U-Net")

