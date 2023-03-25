from dataset.dsets import *
from train_transunet import *

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TTPipeline:
    def __init__(self, dataset_path, model_path, device):
        self.device = device
        self.dataset_path = dataset_path
        self.model_path = model_path

        self.dataframe = load_dataframe(self.dataset_path, cfg.debug)
        # Splitting the dataframe into train/val/test
        self.train_df, self.val_df, self.test_df = self.__train_test_val_split()
        
        
        # Loading the dataframes into datasets of class DukePeopleDataset
        self.train_dataset, self.val_dataset, self.test_dataset = self.__dataset()
        # Loading the datasets into dataloaders for training process
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.__dataloader()

        # If working with large models, one might want to release memmory to increase performance
        if cfg.release_memory:
            del self.train_df, self.val_df, self.test_df
            del self.train_dataset, self.val_dataset, self.test_dataset
    
        self.transunet = TransUNetSegmentation(device)
    def train(self):
        for epoch in range(cfg.epoch):
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                train_loss = self.__loop(self.train_dataloader,  self.transunet.train_step, tepoch)

                val_loss = self.__loop(self.val_dataloader,  self.transunet.test_step, tepoch)
                val_iou = self.transunet.iou(self.val_dataloader)

                print("Epoch [%d]" % (epoch))
                print("\nMean Loss DICE on train:", train_loss)
                print("\nVal Mean Loss DICE on train:", val_loss)
                print("\nVal IOU:", val_iou)


    def __loop(self, loader, step_function, tepoch):
        total_loss = 0
        loss_ = 0
        for i_step, (img, mask) in enumerate(loader):
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_function(img=img, mask=mask)

            total_loss += loss/cfg.batch_size
            
            tepoch.set_postfix(loss=total_loss)
            tepoch.update()
        
        # return torch.tensor(loss).detach().cpu().numpy().mean()
        return total_loss


    def __dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                        batch_size = cfg.batch_size,
                                        num_workers = cfg.num_workers,
                                        shuffle = True)

        val_dataloader = DataLoader(self.val_dataset,
                                        batch_size = cfg.batch_size,
                                        num_workers = cfg.num_workers,
                                        shuffle = True)    

        test_dataloader = DataLoader(self.test_dataset,
                                        batch_size = cfg.batch_size,
                                        num_workers = cfg.num_workers,
                                        shuffle = True)
        
        return train_dataloader, val_dataloader, test_dataloader

    def __dataset(self):
        train_dataset = DukePeopleDataset(df = self.train_df, 
                                        img_w = cfg.img_size_w,
                                        img_h = cfg.img_size_h, 
                                        state = cfg.train_state)

        val_dataset = DukePeopleDataset(df = self.val_df, 
                                        img_w = cfg.img_size_w,
                                        img_h = cfg.img_size_h, 
                                        state = cfg.test_val_state)

        test_dataset = DukePeopleDataset(df = self.test_df, 
                                        img_w = cfg.img_size_w,
                                        img_h = cfg.img_size_h, 
                                        state = cfg.test_val_state) 

        return train_dataset, val_dataset, test_dataset

    def __train_test_val_split(self):
        temp_df, test_df = train_test_split(self.dataframe, 
                                            stratify=self.dataframe.diagnosis, 
                                            test_size=0.2, 
                                            )
    
        temp_df = temp_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)       

        train_df, val_df = train_test_split(temp_df, 
                                            stratify=temp_df.diagnosis, 
                                            test_size=cfg.train_val_split, 
                                            )



        val_df = val_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        return train_df, val_df, test_df