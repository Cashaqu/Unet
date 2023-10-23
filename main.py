import pathlib

from data_preprocess.bubbles_generator import create_project_structure, fill_dataset
from UNet.model import UNet
from UNet.train_function import training
from data_preprocess.dataset_utils import to_loader
from UNet.inference import UNet_eval
import torch
import torch.nn as nn
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    parser.add_argument("--dataset_size",
                        default=100,
                        type=int,
                        help="dataset size")

    parser.add_argument("--coef_split",
                        default=0.8,
                        type=float,
                        help="split the dataset in the specified proportion")

    parser.add_argument("--num_epoch",
                        default=20,
                        type=int,
                        help="number of epochs")

    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="size of batch for train and validation")

    parser.add_argument("--mode",
                        default='inference',
                        type=str,
                        help="train or inference")

    parser.add_argument("--test_size",
                        default=10,
                        type=int,
                        help="number pictures for inference")



    args = parser.parse_args()

    DATASET_SIZE = args.dataset_size
    COEF_SPLIT = args.coef_split
    NUM_EPOCHS = args.num_epoch
    BATCH_SIZE = args.batch_size
    MODE = args.mode
    TEST_SIZE = args.test_size

    if MODE == 'train':

        # create dataset
        create_project_structure()
        fill_dataset(DATASET_SIZE, COEF_SPLIT)

        # prepare dataset
        train_loader = to_loader('./data/X_train/', './data/y_train/', batch_size=BATCH_SIZE)
        valid_loader = to_loader('./data/X_valid/', './data/y_valid/', batch_size=BATCH_SIZE)

        # init model
        model = UNet().to(device)
        last_model = training(
            model=model,
            num_epochs=NUM_EPOCHS,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.005)
        )

        UNet_eval(
            path_to_model = f'./models/model_{last_model}.pt',
            test_size = 10
                )
    else:

        UNet_eval(
            path_to_model='./models/best_model.pt',
            test_size=TEST_SIZE
        )