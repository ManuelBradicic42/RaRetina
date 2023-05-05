from sklearn.model_selection import train_test_split

from dataset.dsets import *
from config import *
from archive.train import dice_loss
import logging
import torch.nn as nn
from torch.optim import SGD
from transunet_control import TransUNetControl
import numpy as np

class TrainTestPipeline:
    def __init__(self, mode, model_name, wd, device):
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.workingdir = wd

        # Creating dataframes 
        self.dataframe = load_dataframe(cfg.DATA_PATH, cfg.DEBUG)
        self.train_df, self.val_df, self.test_df = self.__train_val_test_split(self.dataframe)
        
        # Creating dataloaders for train/test/validation
        self.train_loader = self.__data_loader(self.__data_set(self.train_df, "train"))
        self.test_loader = self.__data_loader(self.__data_set(self.test_df, "test"))
        self.val_loader = self.__data_loader(self.__data_set(self.val_df, "test"))

        # Fetching the model [transunet, resnetunet, unet]
        self.model_control = self.__fetch_model_control(model_name)
        self.model_control.store_info()


    def train(self,):
        train_loss_history = []
        val_loss_history = []
        train_loss_metric_history = []
        val_loss_metric_history = []
        best_loss = 1

        for epoch in range(cfg.epochs):
            print(f"Epoch: {epoch+1}")
            train_loss, train_metric = self.__loop(self.train_loader, self.model_control.train_step)
            train_loss_history.append(train_loss)
            train_loss_metric_history.append(train_metric)

            val_loss, val_metric = self.__loop(self.val_loader, self.model_control.test_step)
            val_loss_history.append(val_loss)
            val_loss_metric_history.append(val_loss)

            result = f"Loss train: {round(train_loss,4)}\nLoss validation: {round(val_loss,4)}\nDice metric train: {round(train_metric,4)}\nDice metric validation: {round(val_metric,4)}"
            print(result)
            logging.info(f"EPOCH RESULTS\nEpoch: {epoch+1}\n{result}")

            if  val_loss < best_loss:
                path = f"{self.workingdir}/model_{self.model_name}_epoch_{epoch}_val_dice_{round(val_metric,4)}.pth"
                self.model_control.save_model(path)
                logging.info(f"SAVING MODEL\nSaving best model with dice_metric_validation:{round(val_metric,4)}")
                print("Saving best model...")
                best_loss = val_loss

        test_loss, test_metric = self.__loop(self.test_loader, self.model_control.test_step)
        logging.info(f"TEST EVALUATION RESULTS\n Loss test: {test_loss} Dice metric validation: {test_metric}")



    def __loop(self, loader, step_function):
        total_loss = []
        total_loss_metric = []

        for i_step, (data, target) in enumerate(loader):
            data = data.to(self.device)
            target = target.to(self.device)

            loss, loss_metric, pred_ = step_function(data=data, target=target)
            
            total_loss_metric.append(loss_metric)
            total_loss.append(loss)
        return np.array(total_loss).mean(), np.array(total_loss_metric).mean()



    def __fetch_model_control(self, model_name):
        print(model_name)
        if model_name == 'transunet':
            logging.info('Loading transunet model')
            model_control = TransUNetControl(self.device, len(self.train_loader))

        if model_name == 'resnetunet':
            logging.info('Loading resnetunet model')

        return model_control
    def __train_val_test_split(self, dataframe):

        #Spliting df into test and train sets
        train_df, test_df = train_test_split(dataframe, stratify=dataframe.diagnosis, test_size=0.2, random_state=42)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        #Splitting train_df into train_df and val_df
        train_df, val_df = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.2, random_state=42)

        train_df = train_df.reset_index(drop=True)

        logging.info(f"LOADING DATALOADER... \nTrain: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")

        return train_df, val_df, test_df 

    def __data_set(self, dataframe, state):
        return DukePeopleDataset(df=dataframe,
                                img_w=cfg.IMG_SIZE_W,
                                img_h=cfg.IMG_SIZE_H,
                                state=state)

    def __data_loader(self, dataset):
        return DataLoader(dataset, 
                        batch_size=cfg.batch_size, 
                        num_workers=cfg.num_workers, 
                        shuffle = True)