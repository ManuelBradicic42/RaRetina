import torch
import sys
import argparse

from train import *
def main_pipeline(parser):
    # (1) Selecting specific GPU; (2) Selecting multiple GPUs as working units
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if parser.mode == 'train':
        ttp = TTPipeline(
            train_path = parser.train_path,
            test_path = parser.test_path,
            model_path = parser.model_path,
            device = device
        )

        ttp.train()

    elif parser.mode == "evaluate":
        ttp

    elif parser.mode == "simulate":
        ttp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'simulate'])
    parser.add_argument('--model_path', required=True, type=str, default=None)
    parser.add_argument('--train_path', required='train' in sys.argv, type=str, default=None)
    parser.add_argument('--test_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--image_path', required='simulate' in sys.argv, type=str, default=None)
    
    parser = parser.parse_args()

    # main_pipeline(