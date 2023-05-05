import torch
import argparse
from train_pipeline import TrainTestPipeline
import logging
import time
from datetime import datetime
import os

def main(parser):
    directory = __create_dir()
    logging.basicConfig(filename=f'{directory}/info.log', level=logging.INFO)
    logging.info('Started')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        logging.info(f'Device unit: {device}')
    else:
        logging.warning(f'Device unit is not cuda, using {device}')

    if parser.mode == 'train':
        pipeline = TrainTestPipeline(parser.mode, parser.model_name, directory, device)

        pipeline.train()


def __create_dir():
    pwd = os.getcwd()
    print(pwd)
    dt_string = "training/raretina-" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    os.mkdir(dt_string)

    return dt_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train','inference'])
    parser.add_argument('--model_name', type=str, required=True, choices=['transunet', 'resnetunet', 'unet'])
    parser = parser.parse_args()

    main(parser)
