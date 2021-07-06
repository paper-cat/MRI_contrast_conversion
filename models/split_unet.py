from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation
from tensorflow.keras.layers import Layer

import tensorflow as tf

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class ConvBlock(Layer):
    def __init__(self, filter_size, dropout_rate):
        super(ConvBlock, self).__init__()

        self.filter_size = filter_size
        # self.dropout_rate = dropout_rate

        self.conv1 = Conv2D(self.filter_size, (3, 3), kernel_initializer='he_normal', padding='same')
        self.batch_norm1 = BatchNormalization()
        self.activate1 = Activation('relu')
        # self.dropout = Dropout(self.dropout_rate)
        self.conv2 = Conv2D(self.filter_size, (3, 3), kernel_initializer='he_normal', padding='same')
        self.batch_norm2 = BatchNormalization()
        self.activate2 = Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.activate1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activate2(x)

        return x


class Attention_gate(Layer):
    def __init__(self, f_g, f_l, f_int):
        super(Attention_gate, self).__init__()

        self.w_g = tf.keras.Sequential(
            [
                Conv2D(f_g, kernel_size=1, strides=1, padding='valid'),
                BatchNormalization()
            ]
        )

        self.w_x = tf.keras.Sequential(
            [
                Conv2D(f_l, kernel_size=1, strides=1, padding='valid'),
                BatchNormalization()
            ]
        )

        self.relu = Activation('relu')

        self.psi = tf.keras.Sequential(
            [
                Conv2D(1, 1, 1, 'valid'),
                BatchNormalization(),
                Activation('sigmoid')
            ]
        )

    def call(self, input_g, input_x):
        g = self.w_g(input_g)
        x = self.w_x(input_x)

        sumed = self.relu(g + x)
        out = self.psi(sumed)

        return x * out


class UnetBlock_origin(Layer):

    def __init__(self):
        super(UnetBlock_origin, self).__init__()

        self.conv_block_in_1 = ConvBlock(64, 0.1)
        self.max_pooling_1 = MaxPooling2D()

        self.conv_block_in_2 = ConvBlock(128, 0.1)
        self.max_pooling_2 = MaxPooling2D()

        self.conv_block_in_3 = ConvBlock(256, 0.1)
        self.max_pooling_3 = MaxPooling2D()

        self.conv_block_in_4 = ConvBlock(512, 0.1)
        self.max_pooling_4 = MaxPooling2D()

        ################################################

        self.conv_block_deep = ConvBlock(1024, 0.1)

        ################################################

        self.up_conv_1 = UpSampling2D((2, 2))
        self.conv_block_out_1 = ConvBlock(512, 0.1)
        self.att1 = Attention_gate(512, 512, 256)

        self.up_conv_2 = UpSampling2D((2, 2))
        self.conv_block_out_2 = ConvBlock(256, 0.1)
        self.att2 = Attention_gate(256, 256, 128)

        self.up_conv_3 = UpSampling2D((2, 2))
        self.conv_block_out_3 = ConvBlock(128, 0.1)
        self.att3 = Attention_gate(128, 128, 64)

        self.up_conv_4 = UpSampling2D((2, 2))
        self.conv_block_out_4 = ConvBlock(64, 0.1)
        self.att4 = Attention_gate(64, 64, 32)

        self.out = Conv2D(1, (1, 1), activation='relu')

    @tf.function
    def call(self, x):
        x1 = self.conv_block_in_1(x)
        x = self.max_pooling_1(x1)

        x2 = self.conv_block_in_2(x)
        x = self.max_pooling_2(x2)

        x3 = self.conv_block_in_3(x)
        x = self.max_pooling_3(x3)

        x4 = self.conv_block_in_4(x)
        x = self.max_pooling_4(x4)

        x = self.conv_block_deep(x)

        x = self.up_conv_1(x)
        x4 = self.att1(x, x4)
        x = tf.concat([x4, x], axis=3)
        x = self.conv_block_out_1(x)

        x = self.up_conv_2(x)
        x3 = self.att2(x, x3)
        x = tf.concat([x3, x], axis=3)
        x = self.conv_block_out_2(x)

        x = self.up_conv_3(x)
        x2 = self.att3(x2, x)
        x = tf.concat([x2, x], axis=3)
        x = self.conv_block_out_3(x)

        x = self.up_conv_4(x)
        x1 = self.att4(x1, x)
        x = tf.concat([x1, x], axis=3)
        x = self.conv_block_out_4(x)

        x = self.out(x)
        return x


class UnetBlock(Layer):

    def __init__(self):
        super(UnetBlock, self).__init__()

        self.conv_block_in_1 = ConvBlock(32, 0.1)
        self.max_pooling_1 = MaxPooling2D()

        self.conv_block_in_2 = ConvBlock(64, 0.1)
        self.max_pooling_2 = MaxPooling2D()

        self.conv_block_in_3 = ConvBlock(128, 0.1)
        self.max_pooling_3 = MaxPooling2D()

        self.conv_block_in_4 = ConvBlock(256, 0.1)
        self.max_pooling_4 = MaxPooling2D()

        ################################################

        self.conv_block_deep = ConvBlock(512, 0.1)

        ################################################

        self.up_conv_1 = UpSampling2D((2, 2))
        self.conv_block_out_1 = ConvBlock(256, 0.1)
        self.att1 = Attention_gate(256, 256, 128)

        self.up_conv_2 = UpSampling2D((2, 2))
        self.conv_block_out_2 = ConvBlock(128, 0.1)
        self.att2 = Attention_gate(128, 128, 64)

        self.up_conv_3 = UpSampling2D((2, 2))
        self.conv_block_out_3 = ConvBlock(64, 0.1)
        self.att3 = Attention_gate(64, 64, 32)

        self.up_conv_4 = UpSampling2D((2, 2))
        self.conv_block_out_4 = ConvBlock(32, 0.1)
        self.att4 = Attention_gate(32, 32, 16)

        self.out = Conv2D(1, (1, 1), activation='relu')

    @tf.function
    def call(self, x):
        x1 = self.conv_block_in_1(x)
        x = self.max_pooling_1(x1)

        x2 = self.conv_block_in_2(x)
        x = self.max_pooling_2(x2)

        x3 = self.conv_block_in_3(x)
        x = self.max_pooling_3(x3)

        x4 = self.conv_block_in_4(x)
        x = self.max_pooling_4(x4)

        x = self.conv_block_deep(x)

        x = self.up_conv_1(x)
        x4 = self.att1(x, x4)
        x = tf.concat([x4, x], axis=3)
        x = self.conv_block_out_1(x)

        x = self.up_conv_2(x)
        x3 = self.att2(x, x3)
        x = tf.concat([x3, x], axis=3)
        x = self.conv_block_out_2(x)

        x = self.up_conv_3(x)
        x2 = self.att3(x2, x)
        x = tf.concat([x2, x], axis=3)
        x = self.conv_block_out_3(x)

        x = self.up_conv_4(x)
        x1 = self.att4(x1, x)
        x = tf.concat([x1, x], axis=3)
        x = self.conv_block_out_4(x)

        x = self.out(x)
        return x


class AttUnetSplit(Model):

    def __init__(self):
        super(AttUnetSplit, self).__init__()

        self.unet_1 = UnetBlock_origin()
        self.unet_2 = UnetBlock_origin()
        # self.unet_3 = UnetBlock_origin()

        self.fcn1 = Dense(256, activation='relu')
        self.out_conv = Conv2D(1, (1, 1), activation='relu')

    def get_config(self):
        pass

    @tf.function
    def call(self, x):
        print(tf.expand_dims(x[:, :, :, 0], axis=3).shape)
        x1 = self.unet_1(tf.expand_dims(x[:, :, :, 0], axis=3))
        x2 = self.unet_2(tf.expand_dims(x[:, :, :, 1], axis=3))
        # x3 = self.unet_3(tf.expand_dims(x[:, :, :, 2], axis=3))

        # x_residual = tf.concat([x1, x2, x3], axis=3)
        x_residual = tf.concat([x1, x2], axis=3)

        x = self.fcn1(x_residual)
        x = Dropout(0.3)(x)
        x = self.out_conv(x)

        return x

    def learning(self, x, y, test_x, test_y):
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy',
                     metrics=[ssim_metric(), psnr_metric()])
        # self.build((None, 384, 384, 3))
        # self.summary()

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='weights/unet_split.ckpt',
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         monitor='val_loss',
                                                         save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=20, verbose=0,
            mode='auto', baseline=None, restore_best_weights=False,
        )

        self.fit(x, y, epochs=1000, validation_data=(test_x, test_y), batch_size=1,
                 callbacks=[cp_callback, early_stopping])


class ssim_metric(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(ssim_metric, self).__init__(**kwargs)
        self.ssim = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ssim = tf.image.ssim(y_true, y_pred, 1)

    def result(self):
        return self.ssim


class psnr_metric(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(psnr_metric, self).__init__(**kwargs)
        self.psnr = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.psnr = tf.image.psnr(y_true, y_pred, 1)

    def result(self):
        return self.psnr
