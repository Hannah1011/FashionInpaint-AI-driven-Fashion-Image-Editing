import cv2
import numpy as np
import random
import os
import torch
from PIL import Image

def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    #print("=====================scale=======================")
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[2] * scale))
    # OpenCV는 (높이, 너비, 채널) 형식을 사용하므로, 채널을 마지막으로 이동
    # 이미지 차원 확인 및 조건부 처리
    if img_temp.shape[0] == 1:  # 단일 채널 처리
        # OpenCV는 단일 채널 이미지를 (높이, 너비) 형식으로 요구
        img_resized = cv2.resize(img_temp[0], new_dims, interpolation=cv2.INTER_LINEAR)
        # 스케일링 후 다시 차원 추가
        scaled_img = np.expand_dims(img_resized, axis=0)
    else:  # 다중 채널 처리
        img_transposed = img_temp.transpose(1, 2, 0)
        scaled_img = cv2.resize(img_transposed, new_dims, interpolation=cv2.INTER_LINEAR)
        scaled_img = scaled_img.transpose(2, 0, 1)
    #print("=====================scale fin=======================")
    return scaled_img


def get_data(img, gt, scale_factor=1.3):
    scale = random.uniform(0.5, scale_factor)
    flip_p = random.uniform(0, 1)

    images = img.astype(float)
    images = scale_im(images, scale)
    images = flip(images, flip_p)
    images = images[np.newaxis, :, :, :]
    images = torch.from_numpy(images.copy()).float()

    # ground truth 처리
    gt = gt.astype(float)
    gt[gt == 255] = 0
    gt = flip(gt, flip_p)
    gt = scale_im(gt, scale)
    labels = gt.copy()
    return images, labels

##########################################################################################################################################
'''
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    # Calculate the scaling factor based on the kernel size
    factor = (kernel_size + 1) // 2

    # Determine the center of the kernel based on whether the kernel size is odd or even
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    # Create an open grid of kernel size dimensions
    og = np.ogrid[:kernel_size, :kernel_size]

    # Calculate the bilinear filter using the open grid and scaling factor
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # Initialize the weight matrix with zeros and set the diagonal elements to the calculated filter
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    # Convert the NumPy array to a PyTorch tensor and return it
    return torch.from_numpy(weight).float()
'''
'''
def segmentation_output(mask, num_classes=7):
    label_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (256, 0, 0), (0, 0, 256), (0, 0, 0), (0, 0, 0)]
    # 0: 전신 1:머리카락 2: 머리~목 3: 상의 4:바지 5: 배경 6: 팔

    h, w = mask.shape

    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    output = np.array(img)

    return output

def segmentation_output(mask, num_classes=7):
    label_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0),(256, 256, 256), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # 0: 전신 1:머리카락 2: 머리~목 3: 상의 4:바지 5: 배경 6: 팔

    h, w = mask.shape

    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    output = np.array(img)

    return output
'''
