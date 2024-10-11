import os
import logging
import glob
import time
import math
import shutil
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging_helper as logging_helper
from utils import *
from network import *
from train_params import parse
import gc
Dr = 114
Dr_7T = 266

args = parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

createDirectory(args.checkpoint_path)
createDirectory(args.checkpoint_path+ 'Results')

### Logger setting ###
logger = logging.getLogger("module.train")
logger.setLevel(logging.INFO)
logging_helper.setup(args.checkpoint_path + 'Results','log.txt')

nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
nowTime = datetime.datetime.now().strftime('%H:%M:%S')
logger.info(f'Date: {nowDate}  {nowTime}')

for key, value in vars(args).items():
    logger.info('{:15s}: {}'.format(key,value))

### Random seed ###
os.environ['PYTHONHASHargs.seed'] = str()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.random.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

### Network & Data loader setting ###
model = R2convNet(channel_in=args.channel_in, kernel_size=args.kernel_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98, last_epoch=-1)

train_set = train_dataset(args.train_path, args.output_map)
valid_set = valid_dataset(args.valid_path, args.output_map)

train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 64)
valid_loader = DataLoader(valid_set, batch_size = args.batch_size, shuffle = False, num_workers = 64)

logger.info(f'Num of batches: {len(train_set)}')

step = 0
train_loss = []
valid_loss = []
nrmse = []
psnr = []
ssim = []
best_loss = math.inf; best_nrmse = math.inf; best_psnr = -math.inf; best_ssim = -math.inf;
best_epoch_loss = 0; best_epoch_nrmse = 0; best_epoch_psnr = 0; best_epoch_ssim = 0;

start_time = time.time()

logger.info("------ Training is started ------")
for epoch in tqdm(range(args.train_epoch), desc = 'EPOCH', position=0):
    ### Training ###
    epoch_time = time.time()
    
    train_loss_list = []
    train_l1loss_list =[]
    train_gdloss_list = []
    train_rmlseloss_list = []
    valid_loss_list =[]
    nrmse_list = []
    psnr_list = []
    ssim_list = []
    
    for train_data in tqdm(train_loader):
        model.train()

        index = train_data[0]
        r2star_input_batch = train_data[1].to(device)
        r2prime_batch = train_data[2].to(device)
        m_batch = train_data[3].to(device)

        ### Masking ###
        # dim: [8, 1, 64, 64, 64]
        r2star_input_batch = r2star_input_batch * m_batch
        r2prime_batch =r2prime_batch * m_batch

        # dim: [8, 2, 64, 64, 64]
        input_batch = r2star_input_batch
        #torch.cat((local_f_batch, r2input_batch), 1)
        label_batch = r2prime_batch
        

        pred = model(input_batch)

        loss, l1loss, gdloss, rmlseloss = total_loss(args.output_map, pred, label_batch, input_batch, m_batch, train_set.r2prime_mean, train_set.r2prime_std, args.w_gdloss, args.w_ssimloss, args.w_rmlseloss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        train_loss_list.append(loss.item())
        train_l1loss_list.append(l1loss.item())
        train_gdloss_list.append(gdloss.item())
        train_rmlseloss_list.append(rmlseloss.item())

        del(r2star_input_batch,m_batch, input_batch, label_batch, loss, l1loss, r2prime_batch); torch.cuda.empty_cache();

    logger.info("Train: EPOCH %04d / %04d | LOSS %.6f |  L1LOSS %.6f |  GDLOSS %.6f | RMSLELOSS %.6f  | TIME %.1fsec | LR %.8f"
          %(epoch+1, args.train_epoch, np.mean(train_loss_list), np.mean(train_l1loss_list), np.mean(train_gdloss_list), np.mean(train_rmlseloss_list), time.time() - epoch_time, optimizer.param_groups[0]['lr']))
    
    ### Validation ###
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    
    with torch.no_grad():
        r2input_batch = torch.tensor(valid_set.r2star_7T[np.newaxis, np.newaxis, ...], device=device, dtype=torch.float)
        m_batch = torch.tensor(valid_set.mask[np.newaxis, np.newaxis, ...], device=device, dtype=torch.float)
        r2output_batch = torch.tensor(valid_set.r2prime[np.newaxis, np.newaxis, ...], device=device, dtype=torch.float)

        label_std = valid_set.r2prime_std
        label_mean = valid_set.r2prime_mean

        ### Normalization ###
        r2input_batch = ((r2input_batch.cpu() - valid_set.r2star_7T_mean) / valid_set.r2star_7T_std).to(device)
        r2output_batch = ((r2output_batch.cpu() - label_mean) / label_std).to(device)

        ### Masking ###
        r2input_batch = r2input_batch * m_batch
        r2prime_batch = r2output_batch * m_batch

        input_batch = r2input_batch
        label_batch = r2prime_batch

        pred = model(input_batch)

        pred[:, 0, ...] = pred[:, 0, ...] * m_batch
        label_batch[:, 0, ...] = label_batch[:, 0, ...] * m_batch

        l1loss = l1_loss(pred, label_batch)
        #ssimloss = ssim_loss(pred, label_batch)
        gradloss = grad_loss(pred, label_batch)

        ### De-normalization ###
        pred= (((pred[:, 0, ...].cpu() * label_std) + label_mean)*Dr).to(device)
        label = (((label_batch.cpu() * label_std) + label_mean)*Dr).to(device)

        pred = pred*m_batch
        pred[pred<0] = 0
        label = label*m_batch
        label[label<0] =0


        
        _nrmse = NRMSE(pred, label, m_batch)
        _psnr = PSNR(pred, label, m_batch)
        _ssim = SSIM(pred, label, m_batch)
        loss = l1loss 
        pred = pred.squeeze()
        pred = pred.cpu()
        pred = pred.numpy()
        pred_r2_map = pred
        label = label.squeeze()
        label = label.cpu()
        label = label.numpy()
        label_r2_map = label

        if (epoch+1) % args.save_step == 0:
            scipy.io.savemat( args.checkpoint_path + 'valid_result_' + str(epoch+1) + '.mat',
                             mdict={'label': label_r2_map,
                                    'pred_r2prime': pred_r2_map})


        valid_loss_list.append(loss.item())
        nrmse_list.append(_nrmse)
        psnr_list.append(_psnr)
        ssim_list.append(_ssim)
        del(m_batch, input_batch, label_batch, l1loss); torch.cuda.empty_cache();
            
            
        logger.info("Valid: EPOCH %04d / %04d | LOSS %.6f | NRMSE %.4f | PSNR %.4f\n | SSIM %.4f\n "
              %(epoch+1, args.train_epoch, np.mean(valid_loss_list), np.mean(_nrmse), np.mean(_psnr), np.mean(_ssim)))
        
        train_loss.append(np.mean(train_loss_list))
        valid_loss.append(np.mean(valid_loss_list))
        nrmse.append(np.mean(_nrmse))
        psnr.append(np.mean(_psnr))
        ssim.append(np.mean(_ssim))

        if np.mean(valid_loss_list) < best_loss:
            print("Best loss model updated")
            save_model(epoch+1, model, args.checkpoint_path, 'best_loss')
            best_loss = np.mean(valid_loss_list)
            best_epoch_loss = epoch+1
        if np.mean(_nrmse) < best_nrmse:
            print("Best nrmse model updated")
            save_model(epoch+1, model, args.checkpoint_path, 'best_nrmse')
            best_nrmse = np.mean(_nrmse)
            best_epoch_nrmse = epoch+1
        if np.mean(_psnr) > best_psnr:
            print("Best psnr model updated")
            save_model(epoch+1, model, args.checkpoint_path, 'best_psnr')
            best_psnr = np.mean(_psnr)
            best_epoch_psnr = epoch+1
        if np.mean(_ssim) > best_ssim:
            print("Best ssim model updated")
            save_model(epoch+1, model, args.checkpoint_path, 'best_ssim')
            best_ssim = np.mean(ssim)
            best_epoch_ssim = epoch+1
            

    ### Saving the model ###
    if (epoch+1) % args.save_step == 0:
        save_model(epoch+1, model, args.checkpoint_path, epoch+1)
logger.info("------ Training is finished ------")
logger.info(f'[best epochs]\nLoss: {best_epoch_loss}\nNRMSE: {best_epoch_nrmse}\nPSNR: {best_epoch_psnr}')
logger.info(f'Total training time: {time.time() - start_time}')

### Plotting learning curve & result curves ###
epoch_list = range(1, args.train_epoch + 1)
plt.ylim((0.01, 0.6))
plt.plot(epoch_list, np.array(train_loss), 'y')
plt.title('Train loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.checkpoint_path + "Results/train_loss_graph.png")
plt.clf()

plt.ylim((0.01, 0.04))
plt.plot(epoch_list, np.array(valid_loss), 'c')
plt.title('Valid loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.checkpoint_path + "Results/valid_loss_graph.png")
plt.clf()

plt.plot(epoch_list, np.array(nrmse), 'y')
plt.ylim((0, 100))
plt.title('NRMSE Graph')
plt.xlabel('epoch')
plt.ylabel('NRMSE')
plt.savefig(args.checkpoint_path + "Results/NRMSE_graph.png")
plt.clf()

plt.plot(epoch_list, np.array(psnr), 'y')
plt.ylim((0, 50))
plt.title('PSNR Graph')
plt.xlabel('epoch')
plt.ylabel('PSNR')
plt.savefig(args.checkpoint_path + "Results/PSNR_graph.png")
plt.clf()

plt.plot(epoch_list, np.array(ssim), 'y')
plt.ylim((0.01, 1.0))
plt.title('SSIM Graph')
plt.xlabel('epoch')
plt.ylabel('SSIM')
plt.savefig(args.checkpoint_path + "Results/SSIM_graph.png")
plt.clf()