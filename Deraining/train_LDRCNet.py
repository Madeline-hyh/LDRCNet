import os
from config import Config 
opt = Config('/home/cls2022/hyh/MPRNet/Deraining/training_LDRCNet.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data,get_enhanced_data,get_mytraining_data
from MPRNet import MPRNet
from model.LDRCNet import *
from model.Derain import *
from losses import *
from pytorch_ssim import ssim
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import itertools
######### Set Seeds ###########
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

start_epoch = 1
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, session, 'results')
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, session, 'models')

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = LDRCNet()
model_restoration.cuda()

model_generation = Rain_Generation()
model_generation.cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(itertools.chain(model_restoration.parameters(),model_generation.parameters()), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

best_psnr = 0
best_ssim = 0
best_epoch = 0


######### Resume ###########
if opt.TRAINING.RESUME:
    model_latest_path =  os.path.join(model_dir,"model_latest.pth")
    if os.path.exists(model_latest_path):
        print("Load Latest model!")
        utils.load_checkpoint_derain(model_restoration,model_latest_path)
        utils.load_checkpoint_gen(model_generation,model_latest_path)
        start_epoch = utils.load_start_epoch(model_latest_path) + 1
        checkpoint = torch.load(model_latest_path)
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        best_epoch = checkpoint["best_epoch"]
        utils.load_optim(optimizer, model_latest_path)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)
    model_generation = nn.DataParallel(model_generation, device_ids = device_ids)

######### Loss ###########
criterion_mse = nn.MSELoss().cuda()
criterion_ace = nn.SmoothL1Loss().cuda()
criterion_l1 = nn.L1Loss(size_average=True, reduce=False).cuda()
criterion_fft = FFTLoss().cuda()


######### DataLoaders ###########
# ori_train_dataset = get_mytraining_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
# enhance_dataset = get_enhanced_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
# train_dataset = ConcatDataset([ori_train_dataset, enhance_dataset])
train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)


print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')



for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_derain_loss = 0
    epoch_gen_loss = 0
    train_id = 1

    model_restoration.train()
    model_generation.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None
        for param in model_generation.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored,deg1,deg2,deg3 = model_restoration(input_)
        generated = model_generation(target,deg1,deg2,deg3)
        
        loss_derain = criterion_mse(restored, target) + 0.2 * (1-ssim(restored,target)) + 0.05 * criterion_fft(restored, target)
        loss_gen = criterion_mse(generated, input_) + 0.2 * (1-ssim(generated, input_)) + 0.05 * criterion_fft(generated, input_)
        loss = loss_derain + loss_gen
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        epoch_derain_loss += loss_derain.item()
        epoch_gen_loss += loss_gen.item()

    #### Evaluation ####
    if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        avg_psnr_val = 0.0
        avg_ssim_val = 0.0
        datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
        for dataset in datasets:
            psnr_val_rgb = 0.0
            ssim_val_rgb = 0.0
            val_dir = os.path.join(opt.TRAINING.VAL_DIR, dataset)
            val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
            for ii, data_val in enumerate((val_loader), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored,_,_,_ = model_restoration(input_)
                    psnr_val_rgb += utils.get_psnr(restored,target)
                    ssim_val_rgb += utils.get_ssim(restored,target)
            
            psnr_val_rgb = round((psnr_val_rgb/ii).item(),5)
            ssim_val_rgb = round((ssim_val_rgb/ii).item(),5)
              
            avg_psnr_val += psnr_val_rgb
            avg_ssim_val += ssim_val_rgb
            print("[epoch %d Datatset: %s PSNR: %.4f SSIM: %.4f]" % (epoch, dataset, psnr_val_rgb, ssim_val_rgb))

        avg_psnr_val = round(avg_psnr_val/5,5)
        avg_ssim_val = round(avg_ssim_val/5,5)

        if avg_psnr_val > best_psnr:
            best_psnr = avg_psnr_val
            best_ssim = avg_ssim_val
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'derain_state_dict': model_restoration.state_dict(),
                        'gen_state_dict': model_generation.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'best_psnr':best_psnr,
                        'best_ssim':best_ssim,
                        'best_epoch':best_epoch
                        }, os.path.join(model_dir,"model_best.pth"))
                
        print("[epoch %d AVG_PSNR: %.4f AVG_SSIM: %.4f--- best_epoch %d Best_PSNR %.4f Best_SSIM %.4f]" % (epoch, avg_psnr_val,avg_ssim_val, best_epoch, best_psnr, best_ssim))

       
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tEpoch_Loss: {:.8f}\tSingle_Loss: {:.8f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss,loss.item(),scheduler.get_lr()[0]))
    print("Derain_loss:{:.8f}\tGen_loss:{:.8f}".format(loss_derain.item(),loss_gen.item()))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'derain_state_dict': model_restoration.state_dict(),
                'gen_state_dict': model_generation.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_psnr':best_psnr,
                'best_ssim':best_ssim,
                'best_epoch':best_epoch,
                }, os.path.join(model_dir,"model_latest.pth")) 

