import os
import logging
import glob
import time
import math
import shutil
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import *
from network import *
from train_params import *

GPU_NUM = '2'
OUTPUT_MAP = 'r2p'
TAG = 'nrmse'
SUBJECTS = ['sampledata']
CHECKPOINT_PATH = '../CheckPoint/'

result_file_name = 'R2primenet7T_'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print(f'Checkpoint: {CHECKPOINT_PATH}\n')

model = R2convNet(channel_in=CHANNEL_IN, kernel_size=KERNEL_SIZE).to(device)
load_file_name = CHECKPOINT_PATH + 'r2pnet7T.pth.tar'
checkpoint = torch.load(load_file_name, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

for SUB in SUBJECTS:
    print(f'SUBJECT : {SUB}\n')
    
    TEST_PATH = '../Data/' + SUB + '/'
    RESULT_PATH = CHECKPOINT_PATH + '../Results/'
    createDirectory(RESULT_PATH)

    ### Network & Data loader setting ###


    ##

    gyro = 42.5775e6
    delta_TE = 0.005
    CF = 123177385
    Dr = 114

    test_set = test_dataset(TEST_PATH)


    print("*** Patching start !!! ***")

    mask = test_set.mask
    r2star_7T = test_set.r2star_7T #r2star_ppm

    r2input = ((r2star_7T - test_set.r2star_7T_mean) / test_set.r2star_7T_std)


    ### Masking ###
    r2input = r2input * mask

    origin_mask = mask.copy()
    print('Data size:', np.shape(origin_mask))

    PS = 64
    matrix_size = np.shape(mask)
    patch_num = np.array([np.ceil(matrix_size[0]/ PS), np.ceil(matrix_size[1]/ PS), np.ceil(matrix_size[2]/ PS)], dtype='int')
    padded_matrix_size = patch_num*PS

    input_r2star_map = np.zeros(padded_matrix_size)
    input_r2star_map[0:matrix_size[0], 0:matrix_size[1], 0:matrix_size[2]] = r2input # zero padding
    padded_mask = np.zeros(padded_matrix_size)
    padded_mask[0:matrix_size[0], 0:matrix_size[1], 0:matrix_size[2]] = mask
    pred_r2_map = np.zeros(padded_matrix_size)

    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)];


    #print('Subject name:', subject)
    print('Matrix size:', matrix_size)
    print('Strides:', strides)


    print("------ Testing is started ------")

    with torch.no_grad():

        model.eval()
        valid_loss_list = []
        nrmse_list = []
        psnr_list = []
        ssim_list = []
        time_list = []
        start_time = time.time()


        r2star_batch = torch.tensor(input_r2star_map, device=device, dtype=torch.float)

        m_batch = torch.tensor(padded_mask, device=device, dtype=torch.float)

        r2star_batch = r2star_batch[np.newaxis, np.newaxis, :, :, :] 

        pred = model(r2star_batch)
        pred = pred*m_batch;

        pred = pred.squeeze()
        pred = pred.cpu()
        pred = pred.numpy()

        pred_r2_map = pred


        #normalization

        time_list.append(time.time() - start_time)



        pred_r2 = ((pred_r2_map*test_set.r2prime_std) + test_set.r2prime_mean)

        pred_r2prime = pred_r2[0:matrix_size[0], 0:matrix_size[1], 0:matrix_size[2]]
        pred_r2prime = pred_r2prime*114
        pred_r2prime[pred_r2prime<0]=0

        total_time = np.mean(time_list)

        scipy.io.savemat(RESULT_PATH + SUB + '_' + result_file_name + TAG + '.mat',
                         mdict={'input_r2star': test_set.r2star_7T,
                                'pred_r2prime': pred_r2prime}

        )

print(f'Total inference time: {total_time}')

print("------ Testing is finished ------")




