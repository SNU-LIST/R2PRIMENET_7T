"""
Experiment settings
    INPUT_MAP: input map of the networks. (r2p: R2', r2s: R2*)
"""
GPU_NUM = '0'
SEED = 777
#TRAIN_PATH = '/home/jiye/chi-sep/DL/R2star_to_R2prime/Dataset/'
TRAIN_PATH = '/home/jiye/chi-sep/DL/R2star_to_R2prime/Dataset/'
VALID_PATH = '/home/jiye/chi-sep/DL/R2star_to_R2prime/Dataset/VAL/'
CHECKPOINT_PATH = '/home/jiye/chi-sep/DL/R2star_to_R2prime/CheckPoint/240722_final_v7/'
OUTPUT_MAP = 'r2p'
Dataset_name = '/'
Validset_name = '/'
"""
Physics-parameters
"""
gyro = 42.5775e6
delta_TE = 0.005
CF = 123177385
Dr = 114
Dr_7T = 114

"""
Network-parameters
"""
CHANNEL_IN = 32
KERNEL_SIZE = 3

"""
Hyper-parameters
"""
TRAIN_EPOCH = 40
LEARNING_RATE = 0.001
BATCH_SIZE = 12
SAVE_STEP = 2
W_SSIMLOSS = 0
W_GDLOSS = 0.1
W_RMLSELOSS = 1.2
PRE_NET_CHECKPOINT_PATH = None#'/home/jiye/chi-sep/DL/R2star_to_R2prime/CheckPoint/240229_Dr_patch64_rmsle_2/'
PRE_FILE = None#'best_loss.pth.tar'
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--gpu_num", default=GPU_NUM)
    parser.add_argument("--seed", default=SEED)

    parser.add_argument("--train_path", default=TRAIN_PATH)
    parser.add_argument("--valid_path", default=VALID_PATH)
    parser.add_argument("--checkpoint_path", default=CHECKPOINT_PATH)
    parser.add_argument("--output_map", default=OUTPUT_MAP)

    parser.add_argument("--gyro", default=gyro)
    parser.add_argument("--delta_TE", default=delta_TE)
    parser.add_argument("--CF", default=CF)
    parser.add_argument("--Dr", default=Dr)


    parser.add_argument("--channel_in", default=CHANNEL_IN)
    parser.add_argument("--kernel_size", default=KERNEL_SIZE)
    
    parser.add_argument("--train_epoch", default=TRAIN_EPOCH)
    parser.add_argument("--learning_rate", default=LEARNING_RATE)
    parser.add_argument("--batch_size", default=BATCH_SIZE)
    parser.add_argument("--save_step", default=SAVE_STEP)
    parser.add_argument("--w_ssimloss", default=W_SSIMLOSS)
    parser.add_argument("--w_gdloss", default=W_GDLOSS)
    parser.add_argument("--w_rmlseloss", default = W_RMLSELOSS)
    parser.add_argument("--PRE_NET_CHECKPOINT_PATH", default = PRE_NET_CHECKPOINT_PATH)
    parser.add_argument("--PRE_FILE", default = PRE_FILE)
    args = parser.parse_args()
    return args