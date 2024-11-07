import os, sys
import argparse
import copy
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download





def preprop_for_diffusion(image, vis_output_model):
  # (3, 128, 128)을 (128, 128, 3)으로 변환'
  image_t = image.transpose(2, 0 ,1)
  array_transposed1 = np.transpose(image_t, (1, 2, 0))

  # 시계방향으로 90도 회전 (반시계 방향으로 세 번 회전)
  image1 = np.rot90(array_transposed1, k=3)
  #plt.imshow(image1)
  #plt.show()
  
  # (3, 128, 128)을 (128, 128, 3)으로 변환
  array_transposed2 = vis_output_model

  # 시계방향으로 90도 회전 (반시계 방향으로 세 번 회전)
  mask_image1 = np.rot90(array_transposed2, k=3)
  #plt.imshow(mask_image1)
  #plt.show()

  image1 = image1*256
  image1 = image1.astype(np.uint8)
  mask_image1 = mask_image1.astype(np.uint8)

  image_source_pil = Image.fromarray(image1)
  image_mask_pil = Image.fromarray(mask_image1)

  display(*[image_source_pil, image_mask_pil])

  return image_source_pil, image_mask_pil


def generate_image(image, mask, prompt, negative_prompt, pipe, seed, device):
  # resize for inpainting
  w, h = image.size
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))

  generator = torch.Generator(device).manual_seed(seed)

  # Change vis_output_model to in_mask
  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
  result = result.images[0]

  return result.resize((w, h))