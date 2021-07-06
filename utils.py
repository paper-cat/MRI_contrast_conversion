import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import scipy.io


def load_images(file_path: str):
    train_data = h5py.File(file_path, mode='r')
    x = train_data['input']
    x = np.array(x, dtype=np.float32)
    y = train_data['label']
    y = np.array(y, dtype=np.float32)

    return x, y


def load_test_images(file_path: str):
    # train_data = h5py.File(file_path, mode='r')
    train_data = scipy.io.loadmat(file_path)
    x = train_data['input']
    x = np.array(x, dtype=np.float32)
    return x


def show_image(x):
    img = Image.fromarray(x, 'RGB')
    img.save('show_data/test.png')
    img.show()


def normalize(input_image, val):
    input_image = input_image / val
    return input_image


def normalize_minmax(input_image):
    out = []
    for img in input_image:
        img[:, :, 0] = img[:, :, 0] / np.max(img[:, :, 0])
        img[:, :, 1] = img[:, :, 1] / np.max(img[:, :, 1])
        img[:, :, 2] = img[:, :, 2] / np.max(img[:, :, 2])

        out.append(img)

    return np.array(out)


def normalize_tv(input_images):
    # define custom transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    output = []
    for img in input_images:
        img_tr = transform(img)

        # calculate mean and std
        mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])

        # define custom transform
        # here we are using our calculated
        # mean & std
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # get normalized image
        img_normalized = transform_norm(img)

        # convert normalized image to numpy
        # array
        img_np = np.array(img_normalized)
        img_np = np.reshape(img_np, (384, 384, img_np.shape[0]))
        output.append(img_np)

    return np.array(output)


def normalize_torchvision(input_images):
    # define custom transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tr = transform(input_images)

    # calculate mean and std
    mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])

    # define custom transform
    # here we are using our calculated
    # mean & std
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # get normalized image
    img_normalized = transform_norm(input_images)

    # convert normalized image to numpy
    # array
    img_np = np.array(img_normalized)
    img_np = np.reshape(img_np, (384, 384, img_np.shape[0]))

    return img_np


def run_clahe(input_image):
    output = []
    for img in input_image:
        # img = cv2.cvtColor(img[:, :, 0], cv2.CV_8UC1)
        img = img[:, :, :1]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        output.append(img_clahe)

    return output
