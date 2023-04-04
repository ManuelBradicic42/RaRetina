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
        
        if cfg.model_name == "TransUNet":
            self.architecture = TransUNetSegmentation(device, no_epochs = cfg.epoch, len_loader = len(self.train_dataloader))
        elif cfg.model_name == "ResnextUnet":
            print("None")
    def train(self):
        train_loss_history = []
        val_loss_history = []
        train_history = []
        highest_acc = 0
        learning_rate = []
        for epoch in range(cfg.epoch):
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                train_loss = self.__loop(self.train_dataloader,  self.architecture.train_step, tepoch)
                train_loss_history.append(train_loss)
                
                val_loss = self.__loop(self.val_dataloader,  self.architecture.test_step, tepoch)
                val_loss_history.append(val_loss)

                val_metric = self.__loop(self.val_dataloader,  self.architecture.dice_loss_metric_step, tepoch)
                train_history.append(val_metric)
                # val_iou = self.architecture.iou(self.val_dataloader)

                print("Epoch [%d]" % (epoch))
                print("\nMean Loss DICE on train:", train_loss)
                print("\nMean Loss DICE on validation:", val_loss)
                # print("\nVal IOU:", val_iou)
                print("\nVal DICE metric:", val_metric)

                print(self.architecture.scheduler.get_last_lr())
                learning_rate.append(self.architecture.scheduler.get_last_lr())

                if val_metric > highest_acc:
                    path = "%s/%s_%i_epoch_%f_acc.pt" %(cfg.path_save, cfg.time, epoch, round(val_metric, 4))
                    print("Saving model to %s" %(path))
                    torch.save(self.architecture.model, path)
                    highest_acc = val_metric
            self.architecture.scheduler_step()
        return train_loss_history, val_loss_history, train_history, learning_rate


    def load_model(self, path):
        self.architecture.model = torch.load(path).module.to(self.device)
        # self.architecture.model = self.architecture.model.module
        self.architecture.model.eval()

    def __loop(self, loader, step_function, tepoch):
        total_loss = 0
        loss_ = 0
        for i_step, (img, mask) in enumerate(loader):
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_function(img=img, mask=mask)

            total_loss += loss
            # print("loss %f" %(loss))

            tepoch.set_postfix(loss=total_loss/(i_step+1))
            tepoch.update()
        
        # return torch.tensor(loss).detach().cpu().numpy().mean()
        return total_loss/ (i_step)


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


    def visualise_example(self):
        image, target = next(iter(self.train_dataloader))

        model = self.architecture.model.to(self.device)

        pred = model(image.to(self.device))
        pred = pred.detach().cpu().numpy()[:,:,:,:]
        pred = np.transpose(pred, (0,2,3,1))
        pred = cv2.normalize(pred, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # From torch to numpy
        image = np.transpose(image.numpy(), (0,2,3,1))
        target = np.transpose(target.numpy(), (0,2,3,1))

        return image, target, pred


    # def visualise_example(self):
    #     test_sample = self.test_df[self.test_df["diagnosis"] == 1].sample(1).values[0]
    #     image = cv2.resize(cv2.imread(test_sample[0]), (224, 224))

    #     # #mask
    #     # mask = cv2.resize(cv2.imread(test_sample[1]), (224, 224))

    #     # # pred
    #     # pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2)
    #     model = self.transunet.model.to(self.device)
    #     # pred = model(pred.to(self.device))
    #     # pred = pred.detach().cpu().numpy()[0,0,:,:]
        
    #     return test_sample, model

    def dtldr(self,):
        a,b = next(iter(self.train_dataloader))

        return a,b 