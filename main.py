from data_preprocess.bubbles_generator import create_project_structure, fill_dataset
from UNet.model import UNet
from UNet.train_function import training
from data_preprocess.dataset_utils import to_loader
from UNet.evaluation import UNet_eval
import torch
import torch.nn as nn

device='cuda' if torch.cuda.is_available() else 'cpu'

DATASET_SIZE = 1000
COEF_SPLIT = 0.8

NUM_EPOCHS = 20
BATCH_SIZE = 8

if __name__ == '__main__':
    # create dataset
    create_project_structure()
    fill_dataset(DATASET_SIZE, COEF_SPLIT)

    # prepare dataset
    train_loader = to_loader('./data/X_train/', './data/y_train/', batch_size=BATCH_SIZE)
    valid_loader = to_loader('./data/X_valid/', './data/y_valid/', batch_size=BATCH_SIZE)

    # init model
    model = UNet().to(device)
    trained_model = training(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.005)
    )

    UNet_eval(
        model=trained_model,
        test_size = 10
        )