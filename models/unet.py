from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D
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


class Unet(Model):

    def __init__(self):
        super(Unet, self).__init__()

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

        self.up_conv_2 = UpSampling2D((2, 2))
        self.conv_block_out_2 = ConvBlock(256, 0.1)

        self.up_conv_3 = UpSampling2D((2, 2))
        self.conv_block_out_3 = ConvBlock(128, 0.1)

        self.up_conv_4 = UpSampling2D((2, 2))
        self.conv_block_out_4 = ConvBlock(64, 0.1)

        self.out = Conv2D(1, (1, 1), activation='relu')

    def get_config(self):
        pass

    @tf.function
    def call(self, x):
        x1 = self.conv_block_in_1(x)
        x = self.max_pooling_1(x1)
        print(x.shape, '1st pooled')

        x2 = self.conv_block_in_2(x)
        x = self.max_pooling_2(x2)
        print(x.shape, '2st pooled')

        x3 = self.conv_block_in_3(x)
        x = self.max_pooling_3(x3)
        print(x.shape, '3st pooled')

        x4 = self.conv_block_in_4(x)
        x = self.max_pooling_4(x4)
        print(x.shape, '4st pooled')

        x = self.conv_block_deep(x)

        x = self.up_conv_1(x)
        x = tf.concat([x4, x], axis=3)
        x = self.conv_block_out_1(x)
        print(x.shape, '1st concat conv ed')

        x = self.up_conv_2(x)
        x = tf.concat([x3, x], axis=3)
        x = self.conv_block_out_2(x)
        print(x.shape, '2st concat conv ed')

        x = self.up_conv_3(x)
        x = tf.concat([x2, x], axis=3)
        x = self.conv_block_out_3(x)
        print(x.shape, '3st concat conv ed')

        x = self.up_conv_4(x)
        x = tf.concat([x1, x], axis=3)
        x = self.conv_block_out_4(x)
        print(x.shape, '4st concat conv ed')

        x = self.out(x)
        print('Done', x.shape)
        return x

    def learning(self, x, y):
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                     metrics=[ssim_metric(), psnr_metric()])
        self.build((None, 384, 384, 3))
        self.summary()

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='weights/unet.ckpt',
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         monitor='val_loss',
                                                         save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=20, verbose=0,
            mode='auto', baseline=None, restore_best_weights=False,
        )

        self.fit(x, y, epochs=1000, validation_split=0.2, batch_size=1, callbacks=[cp_callback, early_stopping])


class ssim_metric(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(ssim_metric, self).__init__(**kwargs)
        self.ssim = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ssim = tf.image.ssim(y_true * 754, y_pred * 754, 754.0)

    def result(self):
        return self.ssim


class psnr_metric(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(psnr_metric, self).__init__(**kwargs)
        self.psnr = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.psnr = tf.image.psnr(y_true * 754, y_pred * 754, 754.0)

    def result(self):
        return self.psnr
