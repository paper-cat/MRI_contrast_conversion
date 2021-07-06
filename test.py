# from models.split_unet import UnetSplit
import numpy as np
# from models.unet import Unet
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.io

from models.att_unet import AttUnet
from utils import load_images, normalize_tv, normalize, load_test_images

if __name__ == '__main__':
    # U-net 기준 목표 PSNR 26.7, SSIM 0.82
    # input 384 x 384

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    x = load_test_images('2021451128_JongHwiPark_contrastconversion.mat')
    x = normalize(x, 1000)

    model = AttUnet()
    model.load_weights('weights/att_unet_max1000.ckpt')
    prediction = model.predict(x, batch_size=1)
    prediction = prediction * 1000

    image = Image.fromarray((prediction[0, :, :, 0]).astype(np.uint8), mode='L')
    image.save('test.png')
