from __future__ import annotations
from re import L
from typing import Callable, Dict, Tuple, Literal, Union

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tqdm import tqdm
import os


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class TrainError(Exception):
    "encoder in ReduceModel isn't trained yet"
    pass


def get_cuda():
    """get device 
    Returns:
         torch.device: cuda if available else cpu
    """
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def RMSE(y_recon, y):
    return torch.sqrt(torch.mean((y_recon-y)**2))


"standard layers: 1378, 702, 351, 176, 88, 44, 22, 11, 5"


class AE(nn.Module):
    losses = {
        "MSE": nn.MSELoss(),
        "RMSE": RMSE
    }

    def __init__(self, layers: Tuple[int] = (1145, 572, 286, 143, 72, 36, 18, 9, 5), last_activation: Callable = nn.ReLU()):
        """AE model

        Args:
            layers (Tuple[int], optional): sizes of layers. Defaults to (1145, 572, 286, 143, 72, 36, 18, 9, 5).
            last_activation (Callable, optional): last activation function for decoder. Defaults to nn.ReLU().
        """

        super(AE, self).__init__()

        self.layers = layers
        self.encoder = nn.Sequential()
        for i in range(len(layers)-2):
            self.encoder.append(nn.Linear(layers[i], layers[i+1]))
            # url https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
            # this why ReLU
            self.encoder.append(nn.ReLU())

        self.encoder.append(nn.Linear(layers[-2], layers[-1]))

        self.decoder = nn.Sequential()
        for i in range(len(layers)-1, 0, -1):
            self.decoder.append(nn.Linear(layers[i], layers[i-1]))
            self.decoder.append(nn.ReLU())
        self.decoder[-1] = last_activation

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _train(self, train_set: TensorDataset, test_set: TensorDataset,
               epochs: int, lr: float = 1e-5, batch_size: int = 128, loss_func: Literal["MSE", "RMSE"] = "MSE") -> Dict[str, object]:
        """train self

        Args:
            train_set (TensorDataset): train data
            test_set (TensorDataset): test data
            epochs (int): number of epochs
            lr (float, optional): learning rate. Defaults to 1e-5.
            batch_size (int, optional): batch size for DataLoader. Defaults to 128.
            loss_func (Literal["MSE", "RMSE"], optional): Loss function. Defaults to "MSE".

        Returns:
            Dict[str, object]: dict with hyper params and train/test losses  
        """

        # device = torch.device("cuda")
        device = get_cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = self.losses[loss_func]
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        train_loader = DataLoader(train_set, batch_size=batch_size, worker_init_fn=seed_worker,
                                  generator=g)
        val_loader = DataLoader(test_set, batch_size=batch_size, worker_init_fn=seed_worker,
                                generator=g)

        # storage for loss
        train_loss_list = [None]*epochs
        test_loss_list = [None]*epochs

        for epoch in tqdm(range(epochs)):
            self.train()
            train_loss = 0.
            for data, in train_loader:

                inputs = data.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)

                loss = criterion(outputs, inputs)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= train_loader.__len__()

            train_loss_list[epoch] = train_loss
            scheduler.step()
            # Validation loop
            val_loss = 0.
            self.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, = data
                    inputs = inputs.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

                val_loss /= len(val_loader)

                test_loss_list[epoch] = val_loss

        train_results = {"model": "AE",
                         "epochs": epochs,
                         "learning_rate": lr,
                         "batch_size": batch_size,
                         "Loss": loss_func,
                         "Latent_space": self.layers[-1],
                         "train_loss": train_loss_list[-1],
                         "test_loss": test_loss_list[-1],
                         "train_loss_list": train_loss_list,
                         "test_loss_list": test_loss_list}

        print(f'Epoch {epochs}, Train Loss: {train_loss_list[-1]}')
        print(f'Epoch {epochs}, Validation Loss: {test_loss_list[-1]}')

        return train_results

    def _transform(self, x: torch.Tensor) -> np.ndarray:
        """Reduce number of features

        Args:
            x (torch.Tensor): x with shape (input_layer_size, int)

        Returns:
            np.ndarray: reduced x
        """

        return self.encoder(x)


                # * if we what to plot latent space per epoch
                # if epoch % 100 == 0:
                    # plot latent space
                    # x_encoded = self.model.encoder(self.dataset).cpu().detach().numpy()
                    # pd.DataFrame(x_encoded).hist(bins=80)
                    # plt.show()
                    # x_pca = PCA(n_components=2).fit_transform(x_encoded)
                    # x_tsne = TSNE(n_components=2).fit_transform(x_encoded)
                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                    # ax1.scatter(x_pca[:,0], x_pca[:,1])
                    # ax1.set_title("PCA")
                    # ax2.scatter(x_tsne[:,0], x_tsne[:,1])
                    # ax2.set_title("TSNE")
                    # plt.show()

# def loss_function(recon_x, x, mu, logvar, beta: float = 0.5,  recon_loss: Literal["MSE", "BCE"] = "BCE"):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1384))
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD


class VAE(nn.Module):
    recon_losses = {"MSE": nn.MSELoss(),
                    "BCE": F.binary_cross_entropy}

    @classmethod
    def KLD(cls, mu, logvar, beta=0.5):
        return - beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def __init__(self, layers: Tuple[int] = (1145, 572, 286, 143, 72, 36, 18, 9, 55), last_activation: Callable = nn.LeakyReLU()):
        """VAE model

        Args:
            layers (Tuple[int], optional): sizes of layers. Defaults to (1145, 572, 286, 143, 72, 36, 18, 9, 55).
            last_activation (Callable, optional): last activation function for decoder. Defaults to nn.LeakyReLU().
        """
        super(VAE, self).__init__()

        self.layers = layers
        self.encoder = nn.Sequential()
        for i in range(len(layers)-2):
            self.encoder.append(nn.Linear(layers[i], layers[i+1]))
            # url https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
            # this why LeakyReLU
            self.encoder.append(nn.LeakyReLU())

        self.decoder = nn.Sequential()
        for i in range(len(layers)-1, 0, -1):
            self.decoder.append(nn.Linear(layers[i], layers[i-1]))
            self.decoder.append(nn.LeakyReLU())
        self.decoder[-1] = last_activation

        self.fc1 = nn.Linear(layers[-2], layers[-1])
        self.fc2 = nn.Linear(layers[-2], layers[-1])
        self.trained = False

    def encode(self, x):
        h = F.relu(self.encoder(x))
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(self.beta*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.layers[0]))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def _train(self, train_set: TensorDataset, test_set: TensorDataset,
               epochs: int, lr: float = 1e-5, batch_size: int = 128, beta: float = 0.5, loss_func: Literal["MSE", "BCE"] = "BCE") -> Dict[str, object]:
        """train self

        Args:
            train_set (TensorDataset): train data
            test_set (TensorDataset): test data
            epochs (int): number of epochs
            lr (float, optional): learning rate. Defaults to 1e-5.
            batch_size (int, optional): batch size for DataLoader. Defaults to 128.
            loss_func (Literal["MSE", "RMSE"], optional): Loss function. Defaults to "MSE".

        Returns:
            Dict[str, object]: dict with hyper params and train/test losses  
        """

        self.beta = beta
        # device = torch.device("cuda")
        device = get_cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        def criterion(x_recon, x, mu, logvar) -> float:
            return self.recon_losses[loss_func](x_recon, x.view(-1, self.layers[0])) + self.KLD(mu, logvar, beta)

        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_loader = DataLoader(test_set, batch_size=batch_size)

        # storage for loss
        train_loss_list = [None]*epochs
        test_loss_list = [None]*epochs

        for epoch in range(epochs):
            self.train()
            train_loss = 0.
            for data, in train_loader:

                inputs = data.to(device)

                optimizer.zero_grad()

                outputs, mu, logvar = self(inputs)

                loss = criterion(outputs, inputs, mu, logvar)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= train_loader.__len__()

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
            train_loss_list[epoch] = train_loss

            # Validation loop
            val_loss = 0.
            self.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, = data
                    inputs = inputs.to(device)
                    outputs, mu, logvar = self(inputs)
                    loss = criterion(outputs, inputs, mu, logvar)
                    val_loss += loss.item()

                # * if we what to plot latent space per epoch
                # if epoch % 100 == 0:
                    # plot latent space
                    # x_encoded = self.model.encoder(self.dataset).cpu().detach().numpy()
                    # pd.DataFrame(x_encoded).hist(bins=80)
                    # plt.show()
                    # x_pca = PCA(n_components=2).fit_transform(x_encoded)
                    # x_tsne = TSNE(n_components=2).fit_transform(x_encoded)
                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                    # ax1.scatter(x_pca[:,0], x_pca[:,1])
                    # ax1.set_title("PCA")
                    # ax2.scatter(x_tsne[:,0], x_tsne[:,1])
                    # ax2.set_title("TSNE")
                    # plt.show()

                val_loss /= len(val_loader)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
                test_loss_list[epoch] = val_loss

        train_results = {"model": "VAE",
                         "epochs": epochs,
                         "learning_rate": lr,
                         "batch_size": batch_size,
                         "Loss": loss_func+"+KLD",
                         "beta": beta,
                         "train_loss": train_loss_list[-1],
                         "test_loss": test_loss_list[-1],
                         "train_loss_list": train_loss_list,
                         "test_loss_list": test_loss_list}

        return train_results

    def _transform(self, x: torch.Tensor) -> np.ndarray:
        """Reduce number of features

        Args:
            x (torch.Tensor): x with shape (input_layer_size, int)

        Returns:
            np.ndarray: reduced x
        """
        return self.encode(x)[0]


def load_data(scale: Literal["minmax", "normalizer"] = "normalizer") -> Tuple[TensorDataset, TensorDataset]:
    """load qmof dataset

    Args:
        scale (Literal["minmax", "normalizer"], optional): scaler. Defaults to "normalizer".

    Returns:
        Tuple[DataLoader, DataLoader]: train and test TensorDatasets 
    """
    path = __file__.__str__().split('\\')

    for i in range(len(path)-1, -1, -1):
        if path[i] == "phase_transmition_ML":
            break
        else:
            path.pop(i)

    path_train = "/".join(path) + f"/qmof_datasets/small_train.csv"
    path_test = "/".join(path) + f"/qmof_datasets/small_test.csv"
    train = TensorDataset(torch.tensor(pd.read_csv(
        path_train, index_col=0).values, dtype=torch.float32))
    test = TensorDataset(torch.tensor(pd.read_csv(
        path_test, index_col=0).values, dtype=torch.float32))

    return train, test


class ReduceModel:
    device = get_cuda()
    train_set, test_set = load_data(scale="minmax")
    # Device and dataset либо в конструктор запихнуть либо только тут оставить 
    dataset = torch.cat(
        (*train_set.tensors, *test_set.tensors)).to(device=device)

    def check_is_trained(self):
        if not self.trained:
            raise TrainError("model isn't trained yet")

    def __init__(self, model: Literal["AE", "VAE"] = "AE", **params) -> None:
        """model for reducing number of features

        Args:
            model (Literal["AE", "VAE"], optional): neural network model. Defaults to "AE".
            params are params of nn model class

        """

        self.trained = False
        self.device = get_cuda()

        match model:
            case "AE":
                self.model = AE(**params).to(self.device)
            case "VAE":
                self.model = VAE(**params).to(self.device)
            case _:
                raise ValueError(f"no model {model} found")

    def train(self, epochs: int, lr: float = 1e-3, batch_size: int = 128, loss_func: Literal['MSE', 'RMSE', "BCE"] = "MSE", **kwargs) -> None:
        """train encoder 

        Args:
            epochs (int): number of epochs to train 
            lr (float, optional): learning rate. Defaults to 1e-3.
            batch_size (int, optional): batch size for loader. Defaults to 128.
            loss_func (Literal["MSE", "MAE"], optional): Loss function. Defaults to "MSE".
            **kwargs: For special params 
        """

        if self.trained:
            raise TrainError("model is trained")

        self.train_results = self.model._train(self.train_set,
                                               self.test_set,
                                               epochs=epochs,
                                               lr=lr,
                                               batch_size=batch_size,
                                               loss_func=loss_func,
                                               **kwargs)

        self.trained = True

    def plot_loss(self):
        """plot train/test loss vs epochs
        """

        self.check_is_trained()

        plt.plot(self.train_results["train_loss_list"],
                 "g", label="train loss")
        plt.plot(self.train_results["test_loss_list"], "r", label="test loss")
        plt.xlabel("epochs")
        plt.ylabel(self.train_results["Loss"])
        plt.title(", ".join([f"{key}: {self.train_results[key]}" for key in self.train_results if key not in [
                  "train_loss", "test_loss", "train_loss_list", "test_loss_list"]]))
        plt.legend()
        plt.show()

    def transform(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """reducing data

        Args:
            x (Union[np.ndarray, pd.DataFrame]): data to reduce

        Raises:
            ValueError: if self.model isn't trained yet

        Returns:
            np.ndarray: reduced x
        """

        self.check_is_trained()

        x_torch = torch.Tensor(np.array(x)).to(self.device)
        x_reduced = self.model._transform(x_torch)

        return x_reduced.cpu().detach().numpy()
    
    def load_model(self, path: str, device: torch.device = get_cuda()):
        """load model from path

        Args:
            path (str): path to model weights .pkl file
        """
        self.model = torch.load(path,  map_location=device)
        self.trained = True


# if __name__ == "__main__":
#     # (1145, 572, 286, 143, 72, 36, 18, 9, 5)
#     model = ReduceModel(AE())
#     model.train(500, lr=1e-3, batch_size=2048, loss_func="MSE")
#     model.transform(model.train_set.tensors[0])
#     model.plot_loss()
