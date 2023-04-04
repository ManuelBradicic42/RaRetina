from architecture.transunet import TransUNet
from config import *
from utils import *
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

import torch
import torch.nn as nn

class TransUNetSegmentation:
    def __init__(self, device, no_epochs, len_loader):
        self.device = device

        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                    in_channels=cfg.transunet.in_channels,
                    out_channels=cfg.transunet.out_channels,
                    head_num=cfg.transunet.head_num,
                    mlp_dim=cfg.transunet.mlp_dim,
                    block_num=cfg.transunet.block_num,
                    patch_dim=cfg.transunet.patch_dim,
                    class_num=cfg.transunet.class_num)
        
        if cfg.parallel:
            # self.model= nn.DataParallel(self.model, device_ids=[0,1])
            self.model= nn.DataParallel(self.model)

        self.model_name = "TransUnet"
        self.model.to(device)
        self.criterion = dice_loss
        self.criterion_metric = dice_metric
        self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate)

        self.scheduler = OneCycleLR(self.optimizer, 
                   max_lr = 1e-2, # Upper learning rate boundaries in the cycle for each parameter group
                   steps_per_epoch = len_loader, # The number of steps xper epoch to train for.
                   epochs = no_epochs, # The number of epochs to train for.
                   anneal_strategy = 'cos') # Specifies the annealing strategy

    def scheduler_step(self,):
        self.scheduler.step()

 

    def train_step(self, **params):
        # losses = []
        
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(params['img'])
        loss = self.criterion(output, params['mask'])
        # losses.append(loss)
        loss.backward()
        
        self.optimizer.step()

        return loss.item(), output

    def test_step(self, **params):
        # compute_iou(self.model, )
        self.model.eval()

        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])

        return loss.item(), pred_mask

    def dice_loss_metric_step(self, **params):
        # compute_iou(self.model, )
        self.model.eval()

        pred_mask = self.model(params['img'])
        loss = self.criterion_metric(pred_mask, params['mask'])

        return loss.item(), pred_mask

    def iou(self, loader, threshold = 0.3):
        valloss = 0
        with torch.no_grad():
            for i_step, (data, target) in enumerate(loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                outputs = self.model(data)

                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < threshold)] = 0.0
                out_cut[np.nonzero(out_cut >= threshold)] = 1.0
                picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
                valloss += picloss
        
        return valloss / i_step
