from nn_models import Autoencoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import time
import matplotlib.pyplot as plt
from cycler import cycler
from typing import Literal


def set_plot():
    plt.rcParams["axes.facecolor"] = '#0d1117'
    plt.rcParams["figure.facecolor"] = '#0d1117'

    # plt.rcParams['figure.figsize'] = [7.0, 3.0]
    plt.rcParams['figure.dpi'] = 100

    # plt.rcParams["axes.spines.bottom.color"]
    # plt.rcParams["axes.spines.left"] = '#0d1117'
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.edgecolor"] = "#eef7f4"

    plt.rcParams["xtick.color"] = '#eef7f4'
    plt.rcParams["ytick.color"] = '#eef7f4'


    plt.rcParams["axes.labelcolor"] = '#eef7f4'

    plt.rcParams["grid.color"] = '#eef7f4'

    plt.rcParams["legend.frameon"] = False

    plt.rcParams['axes.prop_cycle'] = cycler(color=['g', 'r', 'b', 'y'])


def get_cuda():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

set_plot()

device = get_cuda()

# Define the data

def load_data(scale: Literal["minmax", "normalizer"] = "normalizer", batch_size: int = 4096):
    path_train = f"qmof_datasets/{scale}/train.csv"
    path_test = f"qmof_datasets/{scale}/test.csv"
    train = TensorDataset(torch.tensor(pd.read_csv(path_train, index_col=0).values, dtype=torch.float32))
    test = TensorDataset(torch.tensor(pd.read_csv(path_test, index_col=0).values, dtype=torch.float32))

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader

train_loader, val_loader = load_data(scale = "minmax", batch_size=256)

criterion = nn.MSELoss()

MAE = nn.L1Loss(reduction="mean")

lr=1e-3

def train(epochs):
    model=Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global train_loss_list, train_mae_list, valid_loss_list, valid_mae_list
    train_loss_list = []
    train_mae_list = []
    valid_loss_list = []
    valid_mae_list = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        train_mae = 0.
        for data, in train_loader:

            inputs = data.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            mae = MAE(outputs, inputs)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_mae += mae.item()
        
        train_loss /= train_loader.__len__()
        train_mae /= train_loader.__len__()
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
        print(f'Epoch {epoch+1}, Train MAE : {train_mae}')
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        # Validation loop
        val_loss = 0.
        val_mae = 0.
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                mae = MAE(outputs, inputs)
                val_loss += loss.item()
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        print(f'Epoch {epoch+1}, Validation MAE : {val_mae}')
        valid_loss_list.append(val_loss)
        valid_mae_list.append(val_mae)
    torch.save(model, f"ae_{epochs}_lsp_2.pth")
    #print(model.summary())

def train_scheduler(epochs):
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=130)
    global train_loss_list, train_mae_list, valid_loss_list, valid_mae_list
    train_loss_list = []
    train_mae_list = []
    valid_loss_list = []
    valid_mae_list = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        train_mae = 0.
        for data, in train_loader:

            inputs = data.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            mae = MAE(outputs, inputs)

            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_mae += mae.item()
        
        train_loss /= train_loader.__len__()
        train_mae /= train_loader.__len__()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
        print(f'Epoch {epoch+1}, Train MAE : {train_mae}')
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        # Validation loop
        val_loss = 0.
        val_mae = 0.
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                mae = MAE(outputs, inputs)
                val_loss += loss.item()
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        print(f'Epoch {epoch+1}, Validation MAE : {val_mae}')
        valid_loss_list.append(val_loss)
        valid_mae_list.append(val_mae)
  
epochs = 200

start_train = time.time()
train(epochs)

elapsed_time = time.time() - start_train
print('Execution time of training:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


print(train_loss_list[-1])
print(valid_loss_list[-1])
plt.plot(train_loss_list, label="train")
plt.plot(valid_loss_list, label="valid")
plt.legend(labelcolor='w')
plt.grid()
#plt.ylim(0, 1)
plt.savefig(f"reduce_feature/l_space/ae_{epochs}_lsp_2.png")
plt.show()