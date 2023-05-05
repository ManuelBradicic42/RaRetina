import torch

#Initial variables
# IMG_SIZE_W = 256
# IMG_SIZE_H = 256

#Debug variable makes the code run faster and in the debug mode

# Selecting the GPU as device to work with, #3 RTX 6000
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device

from easydict import EasyDict

cfg = EasyDict()
cfg.DEBUG = False
cfg.DATA_PATH = ['/vol/research/Neurocomp/mb01761/datasets/Duke_uni_complementary_data/Control/', 
            '/vol/research/Neurocomp/mb01761/datasets/Duke_uni_complementary_data/AMD/']
cfg.in_parallel = True

cfg.batch_size = 8
cfg.num_workers = 4
cfg.IMG_SIZE_W = 512
cfg.IMG_SIZE_H = 512

cfg.epochs = 15
cfg.learning_rate = 1e-2
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.transunet = EasyDict()
cfg.transunet.img_dim = cfg.IMG_SIZE_W 
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 512
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 1
