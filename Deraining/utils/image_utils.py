import torch
import numpy as np
import cv2

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

from skimage.metrics import  structural_similarity
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
def get_ssim(input_img , compared_img) :
    '''input and compared should be numpy
    batch_size 512 512 3'''
    if not isinstance(input_img , np.ndarray):
        input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1))

    if not isinstance(compared_img , np.ndarray):
        compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))

    Ssim = structural_similarity(input_img , compared_img ,data_range = 1 ,  multichannel=True)
    return Ssim

def get_psnr(input_img , compared_img) :
    '''input and compared should be numpy
    batch_size 512 512 3'''
    if not type(input_img)==type(np.array([1])):
        input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1))

    if not type(compared_img)==type(np.array([1])):
        compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))
    Psnr = compare_psnr(input_img , compared_img ,data_range = 1)
    return Psnr