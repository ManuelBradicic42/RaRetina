import torch
from easydict import EasyDict

cfg = EasyDict()

cfg.debug = False

# Loader
cfg.seed = 42                   # scki-learn random_state seed for train/test/val split
cfg.train_test_split = 0.3      # Split between test and train
cfg.train_val_split = 0.2       # Split between train and validation
cfg.release_memory = False      # If memory for unused variables needs to be realased

# state used to define data augmentation for particular dataset
cfg.train_state = "train"       
cfg.test_val_state = "test"

cfg.batch_size = 24
cfg.num_workers = 4
cfg.IMG_SIZE_W = 512
cfg.IMG_SIZE_H = 512

# SGD
cfg.epoch = 200
cfg.learning_rate = 1e-2
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.transunet = EasyDict()
cfg.transunet.img_dim = 512
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 512
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 1
