import time
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
import scipy
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

    
parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='', type=str, help='Path to weights')
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint_derain(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['Test1200', 'Rain100L', 'Rain100H', 'Test100', 'Test2800']
#datasets = ['Real300']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir)#, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    result_dir  = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)
    
    all_time =0
    count = 0
    img_multiple_of = 8
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_    = data_test[0].cuda()
            filenames = data_test[1]
            
            #Pad the input if not_multiple_of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-h if h%img_multiple_of!=0 else 0
            padw = W-w if w%img_multiple_of!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            h,w = input_.shape[2], input_.shape[3]
            
            st_time = time.time()
            restored,deg1,deg2,deg3 = model_restoration(input_)
            #restored = model_restoration(input_)
            ed_time=time.time()
            cost_time=ed_time-st_time
            all_time +=cost_time
            count +=1
            restored = torch.clamp(restored,0,1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            
            
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time/(count)))#(ii+1)