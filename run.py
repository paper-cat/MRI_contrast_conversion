from models.split_unet import AttUnetSplit
import numpy as np
from models.unet import Unet
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.att_unet import AttUnet
from utils import load_images, normalize_tv, normalize, run_clahe, normalize_minmax

if __name__ == '__main__':
    # U-net 기준 목표 PSNR 26.7, SSIM 0.82
    # input 384 x 384

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    x, y = load_images('data/contrast_conversion_train_dataset.mat')

    print(np.max(x))  # - 1330
    print(np.max(y))  # - 957

    y = normalize(y, 1000)
    x = normalize_minmax(x)

    train_len = int(len(x) * 0.8)
    test_len = len(x) - train_len

    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]

    model = AttUnet()

    model.load_weights('weights/att_unet_maxmin.ckpt')
    prediction = model.predict(test_x, batch_size=1)

    print('SSIM', ssim(test_y, prediction, multichannel=True))
    print('PSNR', psnr(test_y, prediction))
