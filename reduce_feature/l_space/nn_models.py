import torch.nn as nn
import torch
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1375, 702),
            nn.ReLU(),
            nn.Linear(702, 351),
            nn.ReLU(),
            nn.Linear(351, 176),
            nn.ReLU(),
            nn.Linear(176, 88),
            nn.ReLU(),
            nn.Linear(88, 44),
            nn.ReLU(),
            nn.Linear(44, 22),
            nn.ReLU(),
            nn.Linear(22, 11),
            nn.ReLU(),
            nn.Linear(11, 5),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 11),
            nn.ReLU(),
            nn.Linear(11, 22),
            nn.ReLU(),
            nn.Linear(22, 44),
            nn.ReLU(),
            nn.Linear(44, 88),
            nn.ReLU(),
            nn.Linear(88, 176),
            nn.ReLU(),
            nn.Linear(176, 351),
            nn.ReLU(),
            nn.Linear(351, 702),
            nn.ReLU(),
            nn.Linear(702, 1375),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class Autoencoder_params(nn.Module):
#     def __init__(self, layers: list[int], input_size = 1375):
        
#         super(Autoencoder_params, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, layers[0]),
#             *[nn.ReLU(), nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)], nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(input_size, layers[0]),
#             *[nn.ReLU(), nn.Linear(layers[i], layers[i-1]) for i in range(len(layers), 0, -1)], nn.LeakyReLU()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         # Encoder layers
#         self.layers_encode = nn.Sequential(
#             nn.Linear(1384, 702),
#             nn.ReLU(),
#             nn.Linear(702, 351),
#             nn.ReLU(),
#             nn.Linear(351, 176),
#             nn.ReLU(),
#             nn.Linear(176, 88),
#             nn.ReLU(),
#             nn.Linear(88, 44),
#             nn.ReLU(),
#             nn.Linear(44, 22),
#             nn.ReLU(),
#             nn.Linear(22, 11),
#             nn.ReLU(),
#         )

#         self.fc1 = nn.Linear(11, 5)
#         self.fc2 = nn.Linear(11, 5)

#         # Decoder layers
#         self.layers_decoder = nn.Sequential(
#             # nn.Linear(2, 5),
#             # nn.ReLU(),
#             nn.Linear(5, 11),
#             nn.ReLU(),
#             nn.Linear(11, 22),
#             nn.ReLU(),
#             nn.Linear(22, 44),
#             nn.ReLU(),
#             nn.Linear(44, 88),
#             nn.ReLU(),
#             nn.Linear(88, 176),
#             nn.ReLU(),
#             nn.Linear(176, 351),
#             nn.ReLU(),
#             nn.Linear(351, 702),
#             nn.ReLU(),
#             nn.Linear(702, 1384),
#             nn.Sigmoid()
#         )

#     def encode(self, x):
#         h = F.relu(self.layers_encode(x))
#         return self.fc1(h), self.fc2(h)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         out = self.layers_decoder(z)
#         return out

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 1384))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# class VAE_Dr(nn.Module):
#     def __init__(self, params = [1384, 702, 0.2, 351, 0.2, 176, 0.2, 88, 0.2, 44, 0.2, 22, 0.2, 11]):
#         super(VAE_Dr, self).__init__()
#         # Encoder layers
#         self.layers_encode = nn.Sequential(
#             nn.Linear(params[0], params[1]),
#             nn.ReLU(),
#             nn.Dropout(params[2]),
#             nn.Linear(params[1], params[3]),
#             nn.ReLU(),
#             nn.Dropout(params[4]),
#             nn.Linear(params[3], params[5]),
#             nn.ReLU(),
#             nn.Dropout(params[6]),
#             nn.Linear(params[5], params[7]),
#             nn.ReLU(),
#             nn.Dropout(params[8]),
#             nn.Linear(params[7], params[9]),
#             nn.ReLU(),
#             nn.Dropout(params[10]),
#             nn.Linear(params[9], params[11]),
#             nn.ReLU(),
#             nn.Dropout(params[12]),
#             nn.Linear(params[11], params[13]),
#             nn.ReLU(),
#             # nn.Linear(11, 5),
#             # nn.ReLU()
#         )

#         self.fc1 = nn.Linear(params[13], 5)
#         self.fc2 = nn.Linear(params[13], 5)

#         # Decoder layers
#         self.layers_decoder = nn.Sequential(
#             # nn.Linear(2, 5),
#             # nn.ReLU(),
#             nn.Linear(5, params[13]),
#             nn.ReLU(),
#             nn.Linear(params[13], params[11]),
#             nn.ReLU(),
#             nn.Linear(params[11], params[9]),
#             nn.ReLU(),
#             nn.Linear(params[9], params[7]),
#             nn.ReLU(),
#             nn.Linear(params[7], params[5]),
#             nn.ReLU(),
#             nn.Linear(params[5], params[3]),
#             nn.ReLU(),
#             nn.Linear(params[3], params[1]),
#             nn.ReLU(),
#             nn.Linear(params[1], params[0]),
#             nn.LeakyReLU()
#         )

#     def encode(self, x):
#         h = F.relu(self.layers_encode(x))
#         return self.fc1(h), self.fc2(h)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         out = self.layers_decoder(z)
#         return out

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 1384))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1384), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD

# def loss_function_mse(recon_x, x, mu, logvar):
#     MSE =  F.mse_loss(x, recon_x)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + KLD

def metric(recon_x, x):
    MSE = torch.nn.MSELoss()
    MAE = torch.nn.L1Loss()
    return MSE(recon_x, x.view(-1, 1384)).item(), MAE(recon_x, x.view(-1, 1384)).item()