import os
import math
import numpy as np
import h5py
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import mat73
from math import log10, sqrt
from numpy.fft import fftn, ifftn, fftshift
from skimage.metrics import structural_similarity as ssim
import pytorch_ssim

class train_dataset():
    def __init__(self, path, output_map):
        data_file = h5py.File(path + 'train_patch_arlo_rotaug.hdf5', "r")
        value_file = scipy.io.loadmat(path + 'r2pnet7T_norm_factor.mat')
        
        self.output_map = output_map
        
        self.r2prime = data_file['pR2prime']
        self.r2star_7T = data_file['pR2star_7T']
        self.mask = data_file['pMask']
        
        self.r2prime_mean = value_file['r2prime_mean']
        self.r2prime_std = value_file['r2prime_std']
        
        self.r2star_7T_mean = value_file['r2star_7T_mean']
        self.r2star_7T_std = value_file['r2star_7T_std']
        
        
    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        # dim: [1, 64, 64, 64]
        r2s_7T_batch = torch.tensor(self.r2star_7T[idx], dtype=torch.float).unsqueeze(0)
        r2prime_batch = torch.tensor(self.r2prime[idx], dtype=torch.float).unsqueeze(0)
        m_batch = torch.tensor(self.mask[idx], dtype=torch.float).unsqueeze(0)

        ### Normalization ###
        r2s_7T_batch = ((r2s_7T_batch - self.r2star_7T_mean) / self.r2star_7T_std)
        r2prime_batch = ((r2prime_batch - self.r2prime_mean) / self.r2prime_std)

        return idx, r2s_7T_batch, r2prime_batch, m_batch
        

class valid_dataset():
    def __init__(self, path, output_map):
        data_file = mat73.loadmat(path + '/SUB08/R2star_prime_pair_arlo_crop.mat')
        value_file = scipy.io.loadmat(path + '../../CheckPoint/r2pnet7T_norm_factor.mat')

        ### Converting Hz maps to ppm ###
        CF = 123177385
        Dr = 114

        r2prime = data_file['r2prime']
        r2star_7T = data_file['r2star']
        r2star_7T[r2star_7T<0]=0
        r2prime_in_ppm = r2prime / Dr
        r2star_7T_in_ppm = r2star_7T / Dr_7T
        
        
        self.mask = data_file['Mask']
        self.r2prime = r2prime_in_ppm
        self.r2star_7T = r2star_7T_in_ppm
        self.r2prime_std = value_file['r2prime_std']
        self.r2prime_mean = value_file['r2prime_mean']
        
        self.r2star_7T_std = value_file['r2star_7T_std']
        
        self.r2star_7T_mean = value_file['r2star_7T_mean']
                
        self.matrix_size = self.r2star_7T.shape

class test_dataset():
    def __init__(self, path):

        data_file = mat73.loadmat(path + 'Test.mat')
        value_file = scipy.io.loadmat(path + '../../CheckPoint/r2pnet7T_norm_factor.mat')

        ### Converting Hz maps to ppm ###
        Dr = 114

        r2star_7T = data_file['r2star_7T']
        r2star_7T[r2star_7T<0]=0
        
        
        r2star_7T_in_ppm = r2star_7T / Dr
        
        self.mask = data_file['mask_sharp_12']
        self.r2star_7T = r2star_7T_in_ppm
        
        
        self.r2prime_mean = value_file['r2prime_mean']
        self.r2prime_std = value_file['r2prime_std']
        
        self.r2star_7T_mean = value_file['r2star_7T_mean']
        self.r2star_7T_std = value_file['r2star_7T_std']

        
        self.matrix_size = self.mask.shape
        
                
def Concat(x, y):
    return torch.cat((x,y),1)


class Conv3d(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=k_size, stride=1, padding=int(k_size/2), dilation=1)
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    
class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.conv=nn.Conv3d(c_in,  c_out, kernel_size=1, stride=1, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self,x):
        return self.conv(x)
    
    
class Pool3d(nn.Module):
    def __init__(self):
        super(Pool3d, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1)
    
    def forward(self,x):
        return self.pool(x)
    
    
class Deconv3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(Deconv3d, self).__init__()
        self.deconv=nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.deconv.weight)
    
    def forward(self,x):
        return self.deconv(x)



# class Conv2d(nn.Module):
#     def __init__(self, c_in, c_out, k_size):
#         super(Conv2d, self).__init__()
#         self.conv = nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=1, padding=int(k_size/2), dilation=1)
#         self.bn = nn.BatchNorm2d(c_out)
#         self.act = nn.ReLU()
#         nn.init.xavier_uniform_(self.conv.weight)
        
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

    
# class Conv2(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(Conv2, self).__init__()
#         self.conv=nn.Conv2d(c_in,  c_out, kernel_size=1, stride=1, padding=0, dilation=1)
#         nn.init.xavier_uniform_(self.conv.weight)
    
#     def forward(self,x):
#         return self.conv(x)
    
    
# class Pool2d(nn.Module):
#     def __init__(self):
#         super(Pool2d, self).__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
    
#     def forward(self,x):
#         return self.pool(x)
    
    
# class Deconv2d(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(Deconv2d, self).__init__()
#         self.deconv=nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2, padding=0, dilation=1)
#         nn.init.xavier_uniform_(self.deconv.weight)
    
#     def forward(self,x):
#         return self.deconv(x)

    
def l1_loss(x, y):
    return torch.abs(x-y).mean()

def ssim_loss(x, y):
    x = x.squeeze()
    y = y.squeeze()
    
    s_loss  = pytorch_ssim.SSIM(window_size = 11)
    ssim_ = s_loss(x,y)
    s_loss_ = 1 - ssim_
    
    return s_loss_


def mse_loss(x,y):
    return ((x-y)**2).mean()

def rmsle_loss(x,y):
    # x = pred, y = label
    
    x_ = torch.relu(x + 1)
    y_ = torch.relu(y + 1)
    
    log_x = torch.log(x_ + 1e-7)
    log_y = torch.log(y_ + 1e-7)
    #print(log_x)
    #%print(log_y)
    
    return mse_loss(log_x, log_y)



def grad_loss(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(x_cen.shape, device=device)
                else:
                    temp = torch.relu(x_slice-x_cen)/s
                grad_x = grad_x + temp

    y_cen = y[:,:,1:-1,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(y_cen.shape, device=device)
                else:
                    temp = torch.relu(y_slice-y_cen)/s
                grad_y = grad_y + temp

    return l1_loss(grad_x, grad_y)


def total_loss(input_map, p, y, x1, m, r2prime_mean, r2prime_std, w1, w2, w3):
    """
    Args:
        input_map: type of input map of the network. (r2p or r2s)
        p (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. predicted susceptability map.
        y (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. susceptability map (label).
        x1 (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        x2 (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. r2prime map.
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
        w1 (float): weighting factor for model losses
        w2 (float): weighting factor for gradient losses

    Returns:
        l1loss (torch.float): L1 loss. 
        tloss (torch.float): total loss. sum of above three losses with weighting factor
    """
    ### Splitting into positive/negative maps & masking ###

    p_pos = p[:, 0, :, :, :]
    pred_x_pos = p_pos[:, np.newaxis, :, :, :] * m
    
    l_pos = y[:, 0, :, :, :]
    
    label_x_pos = l_pos[:, np.newaxis, :, :, :] * m
    
    pred = pred_x_pos
    label = label_x_pos
    
    r2p = x1 * m
    
    ### L1 loss ###
    l1loss = l1_loss(pred, label)
    gdloss = grad_loss(pred, label)
    ssimloss = ssim_loss(pred, label)
    rmsleloss = rmsle_loss(pred, label)
    
    ### Gradient loss ###
#     gdloss_pos = grad_loss(pred_x_pos, label_x_pos)
#     gdloss_neg = grad_loss(pred_x_neg, label_x_neg)
    
#     gdloss = (gdloss_pos + gdloss_neg)
    
    ### De-normalization ##
    ### Model loss ###
#     mdloss = model_loss(input_map, pred_sus, pred_r2p, local_f, r2p, m, d)
        
#     total_loss = l1loss + mdloss * w1 + gdloss * w2
    total_loss = l1loss + gdloss*w1 + ssimloss * w2 + rmsleloss * w3
    return total_loss, l1loss, gdloss, rmsleloss



def grad_loss_2d(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1]
            s = np.sqrt(i*i+j*j)
            if s == 0:
                temp = torch.zeros(x_cen.shape, device=device)
            else:
                temp = torch.relu(x_slice-x_cen)/s
            grad_x = grad_x + temp

    y_cen = y[:,:,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1]
            s = np.sqrt(i*i+j*j)
            if s == 0:
                temp = torch.zeros(y_cen.shape, device=device)
            else:
                temp = torch.relu(y_slice-y_cen)/s
            grad_y = grad_y + temp

    return l1_loss(grad_x, grad_y)




# def total_loss_2d(input_map, p, y, x1, m, r2prime_mean, r2prime_std, w1):
#     """
#     Args:
#         input_map: type of input map of the network. (r2p or r2s)
#         p (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. predicted susceptability map.
#         y (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. susceptability map (label).
#         x1 (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
#         x2 (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. r2prime map.
#         m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
#         d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
#         w1 (float): weighting factor for model losses
#         w2 (float): weighting factor for gradient losses

#     Returns:
#         l1loss (torch.float): L1 loss. 
#         tloss (torch.float): total loss. sum of above three losses with weighting factor
#     """
#     ### Splitting into positive/negative maps & masking ###
#     #p_pos = p[:, 0, :, :, :]
#     #pred_x_pos = p_pos[:, np.newaxis, :, :, :] * m
#     pred_x_pos = p * m
#     label_x_pos = y * m
# #     l_pos = y[:, 0, :, :, :]
    
# #     label_x_pos = l_pos[:, np.newaxis, :, :, :] * m
    
#     pred = pred_x_pos
#     label = label_x_pos
    
#     r2p = x1 * m
    
#     ### L1 loss ###
#     l1loss = l1_loss(pred, label)
#     gdloss = grad_loss_2d(pred, label)
    
#     ### Gradient loss ###
# #     gdloss_pos = grad_loss(pred_x_pos, label_x_pos)
# #     gdloss_neg = grad_loss(pred_x_neg, label_x_neg)
    
# #     gdloss = (gdloss_pos + gdloss_neg)
    
#     ### De-normalization ##
#     ### Model loss ###
# #     mdloss = model_loss(input_map, pred_sus, pred_r2p, local_f, r2p, m, d)
        
# #     total_loss = l1loss + mdloss * w1 + gdloss * w2
#     #total_loss = l1loss + gdloss*w1
#     total_loss = l1loss + gdloss*w1
#     return total_loss, l1loss

def dipole_kernel(matrix_size, voxel_size, B0_dir):
    """
    Args:
        matrix_size (array_like): should be length of 3.
        voxel_size (array_like): should be length of 3.
        B0_dir (array_like): should be length of 3.
        
    Returns:
        D (ndarray): 3D dipole kernel matrix in Fourier domain.  
    """    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/((X**2 + Y**2 + Z**2) + 1e-6)
    D[np.isnan(D)] = 0
    D = fftshift(D)
    return D


def save_model(epoch, model, PATH, TAG):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict()},
        f'{PATH}/{TAG}.pth.tar')
    torch.save(model, f'{PATH}/model.pt')
    
    
def NRMSE(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask
    mask = mask.bool()
    #mask = mask.astype(bool)

    mse = torch.mean((im1[mask]-im2[mask])**2)

    if torch.mean(im2**2) == 0:
        nrmse = 0
    else :
        #nrmse = sqrt(mse)/sqrt(torch.mean(im2**2))
        nrmse = sqrt(mse) / sqrt(torch.mean(im2[mask]**2))
    return 100*nrmse


def PSNR(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask
    #mask = mask.astype(bool)
    mask = mask.bool()
    #mse = torch.mean(torch.masked_select((im1-im2)**2, mask))
    mse = torch.mean((im1[mask]-im2[mask])**2)

    #mse = torch.mean((im1-im2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = max(im2[mask])
    #PIXEL_MAX = 1
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def SSIM(im1, im2, mask):
    im1 = im1.cpu().detach().numpy(); im2 = im2.cpu().detach().numpy(); mask = mask.cpu().detach().numpy()
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    mask = np.squeeze(mask)
    im1 = im1 * mask; im2 = im2 * mask;
    mask = mask.astype(bool)

#     min_im = np.min([np.min(im1),np.min(im2)])
#     im1[mask] = im1[mask] - min_im
#     im2[mask] = im2[mask] - min_im
    
#     max_im = np.max([np.max(im1),np.max(im2)])
#     im1 = 255*im1/max_im
#     im2 = 255*im2/max_im

    _, ssim_map =ssim(im1, im2, data_range=200, gaussian_weights=True, K1=0.01, K2=0.03, full=True, channel_axis = None)
    return np.mean(ssim_map[mask])


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        