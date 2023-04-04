import torch
from easydict import EasyDict
from time import gmtime, strftime

cfg = EasyDict()
cfg.model_name = "TransUNet"      # ResnextUnet

cfg.debug = True                # if flag then dataset gets reduced to compute faster
cfg.release_memory = False      # If memory for unused variables needs to be realased
cfg.reduced_size = 128

# Paths
cfg.path_save = "/models"
cfg.time = strftime("%Y-%m-%d-%H_%M_%S", gmtime())
cfg.path_save = "/vol/research/Neurocomp/mb01761/git/models"
cfg.dataset_path = ['/vol/research/Neurocomp/mb01761/datasets/Duke_uni_complementary_data/Control/', 
            '/vol/research/Neurocomp/mb01761/datasets/Duke_uni_complementary_data/AMD/']
# Loader
cfg.seed = 42                   # scki-learn random_state seed for train/test/val split
cfg.train_test_split = 0.3      # Split between test and train
cfg.train_val_split = 0.2       # Split between train and validation

# state used to define data augmentation for particular dataset
cfg.train_state = "train"       
cfg.test_val_state = "test"

cfg.batch_size = 2
cfg.num_workers = 4
cfg.img_size_w = 512
cfg.img_size_h = 512
cfg.parallel = True            # If one wants to train the model in parallel vs 1 GPU (local) 

# SGD
cfg.epoch = 120
cfg.learning_rate = 1e-3
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 15
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
