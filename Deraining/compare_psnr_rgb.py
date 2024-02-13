import cv2
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import ToTensor , ToPILImage ,Resize
import matplotlib.pyplot as plt
import copy
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os
from os.path import join
from data_RGB import get_mpr_data
from torch.utils.data import DataLoader
to_tensor = ToTensor()
to_pil = ToPILImage()
resize_512 = Resize((512,512))

def Img2Tensor(path , is_batch = True):
    img=Image.open(path)
    tensor = to_tensor(img)
    if is_batch:
        tensor = torch.unsqueeze(tensor ,dim=0)
    return tensor


def get_psnr_ssim(input_img , compared_img , num=5) :
    """
    input_img & compared_img:tensor
        size : bn 3 w h
    num :float round
    """
    if not type(input_img) == type(np.array([1])):
        input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1))

    if not type(compared_img) == type(np.array([1])):
        compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))

    Ssim = compare_ssim(input_img , compared_img ,data_range = 1 ,  multichannel=True)
    # Ssim = compare_ssim(input_img , compared_img ,data_range = 1)
    Psnr = compare_psnr(input_img , compared_img ,data_range = 1)
    Psnr = round(Psnr.item(), num)
    Ssim = round(Ssim.item(), num)
    return Psnr,Ssim

if __name__=='__main__':
    indir_root='./Datasets/test/'
    outdir_root='./Datasets/results_mprnet/'
    datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
    avg_PSNR =0.0
    avg_SSIM =0.0
    import time
    starttime = time.time()
    for dataset in datasets:
        indir = os.path.join(indir_root,dataset)
        outdir = os.path.join(outdir_root,dataset)
        mpr_dataset = get_mpr_data(indir,outdir, {'patch_size':256})
        val_loader = DataLoader(dataset=mpr_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        test_Psnr_sum = 0.0
        test_Ssim_sum = 0.0
        for ii, data_val in enumerate((val_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            with torch.no_grad():
                Psnr, Ssim= get_psnr_ssim(input_,target)
                test_Psnr_sum += Psnr
                test_Ssim_sum += Ssim
                avg_PSNR += Psnr
                avg_SSIM += Ssim
        print("[Datatset: %s PSNR: %.5f SSIM:%.5f]" % ( dataset, test_Psnr_sum/ii,test_Ssim_sum/ii))
    endtime = time.time()
    print("avg_time:%.5f" %((endtime-starttime)/4300))
    print("[avg_PSNR: %.5f avg_SSIM:%.5f]" % (avg_PSNR/4300,avg_SSIM/4300))