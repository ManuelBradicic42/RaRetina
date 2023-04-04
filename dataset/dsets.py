import cv2
import os
import glob
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import median_filter
from sklearn.utils import shuffle


from config import *


class DukePeopleDataset(Dataset):

    def __init__(self, df, img_w, img_h, state):
        self.IMG_SIZE_W = img_w
        self.IMG_SIZE_H = img_h
        self.df = df
        self.in_channels = 3
        self.out_channels = 1
        
        self.transforms = self.define_transorms(state)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.resize(cv2.imread(self.df.iloc[idx, 0]), (self.IMG_SIZE_W, self.IMG_SIZE_H))
        mask = cv2.resize(cv2.imread(self.df.iloc[idx, 1],0), (self.IMG_SIZE_W, self.IMG_SIZE_H))
#         image = cv2.fastNlMeansDenoising(image,None,15,7,21)
#         image = median_filter(image, [3, 5, 5])

        augmented = self.transforms(image = image,
                                      mask = mask)
        
        image = augmented['image']
        mask = augmented['mask'] / 255.
        mask = mask.unsqueeze(0)
        
        return image, mask

    def get_dataframe(self):
        return self.df

#     def define_transorms(self):
#         transforms = A.Compose([
#             A.HorizontalFlip(p=0.5),
# #             A.GridDistortion(p=1),
#             A.ElasticTransform(p=0.2),
#             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                         A.RandomBrightnessContrast(p=0.2),
#             ToTensorV2(),
#         ])
#         return transforms


    def define_transorms(self, state):
        if state == "train":
            transforms = A.Compose([
#                 A.RandomBrightnessContrast(p=1),
                A.HorizontalFlip(p=0.5),
                # A.ElasticTransform(p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p =1.0),
                ToTensorV2(),
            ])
            return transforms
        if state == "test":
            transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p =1.0),
                ToTensorV2(),
            ])
            return transforms
            

def load_dataframe(DATA_PATH, mode):
    def load_paths(path):  
        temp = [] 
        for dp in path:
            for sub_dir_path in glob.glob(dp + "*"):
                if os.path.isdir(sub_dir_path):
                    temp.append(sub_dir_path)
        return temp
    
    #Sorting the dataframe and separating masks and images
    def sort_dataframe(df):
        #Masks/Not masks
        df_imgs = df[~df['path'].str.contains("label")]
        df_masks = df[df['path'].str.contains("label")]   

        # Data sorting
        imgs = sorted(df_imgs["path"].values)
        masks = sorted(df_masks["path"].values)

        # Merging imgs and masks into one dataframe
        df = pd.DataFrame({"image_path": imgs,
                        "mask_path": masks})
        
        return df

    # Adding a column for diagnosis
    def positive_negative_diagnosis(mask_path):
        if "Control" in mask_path: return 0
        else: return 1


    # Gettings paths 
    paths = load_paths(DATA_PATH)

    data_map = []
    for sub_dir_path in paths:
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
                
    df = pd.DataFrame({"dirname" : data_map[::2],
                    "path" : data_map[1::2]})


    df = sort_dataframe(df)
    df["diagnosis"] = df["mask_path"].apply(lambda m: positive_negative_diagnosis(m))        

    # If debug mode is true, reduce the dataset.
    if mode:
        df = shuffle(df)
        df = df[:cfg.reduced_size]

    return df
