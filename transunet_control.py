import torch
from architecture.transunet import TransUNet
from config import *
from loss import dice_coef_metric,bce_dice_loss, dice_loss
from torch.optim import SGD
import logging
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

class TransUNetControl:
    def __init__(self, device, length):
        self.device = device
        self.criterion = dice_loss

        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                                in_channels=cfg.transunet.in_channels,
                                out_channels=cfg.transunet.out_channels,
                                head_num=cfg.transunet.head_num,
                                mlp_dim=cfg.transunet.mlp_dim,
                                block_num=cfg.transunet.block_num,
                                patch_dim=cfg.transunet.patch_dim,
                                class_num=cfg.transunet.class_num)

        self.optimizer = SGD(self.model.parameters(),
                            lr=cfg.learning_rate,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)     
                               
        # self.scheduler = OneCycleLR(self.optimizer, 
        #                             max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
        #                             epochs = cfg.epochs, # The number of epochs to train for.
        #                             steps_per_epoch = length,
        #                             anneal_strategy = 'cos') # Specifies the annealing strategy

        try:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])
        except:
            # logging.warning("Not managed to used dataparallel!")
            pass

        self.model.to(device)
        logging.info(f"Moving model to {device}")

    def train_step(self, **params):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(params['data'])

        pred_hat = np.copy(pred.data.cpu().numpy())
        pred_hat[np.nonzero(pred_hat < 0.5)] = 0.0
        pred_hat[np.nonzero(pred_hat > 0.5)] = 1.0
        loss_metric = dice_coef_metric(pred_hat, params['target'].cpu().numpy())
        
        loss = self.criterion(pred, params['target'])
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        return loss.item(), loss_metric, pred

    def test_step(self, **params):
        self.model.eval()

        pred = self.model(params['data'])

        pred_hat = np.copy(pred.data.cpu().numpy())
        pred_hat[np.nonzero(pred_hat < 0.5)] = 0.0
        pred_hat[np.nonzero(pred_hat > 0.5)] = 1.0

        loss_metric = dice_coef_metric(pred_hat, params['target'].cpu().numpy())

        loss = self.criterion(pred, params['target'])

        return loss.item(), loss_metric, pred

    # Function for saving models state
    def save_model(self, PATH):
        torch.save(self.model.state_dict(), PATH)

    # Function for loading a pre-trained model
    def load_model(self, PATH):
        self.model.load_state_dict(torch.load(PATH))

    # Functions used for documenting model and optimizer 
    def store_info(self):
        model_info = {'img_dim' : cfg.transunet.img_dim,
            'in_channels' : cfg.transunet.in_channels,
            'out_channels':cfg.transunet.out_channels,
            'head_num':cfg.transunet.head_num,
            'mlp_dim':cfg.transunet.mlp_dim,
            'block_num':cfg.transunet.block_num,
            'patch_dim':cfg.transunet.patch_dim,
            'class_num':cfg.transunet.class_num}

        logging.info(f"MODEL DATA: {model_info}")
        logging.info(f"OPTIMIZER: {self.optimizer}")
