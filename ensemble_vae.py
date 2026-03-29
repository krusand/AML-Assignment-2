# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import pandas as pd
# Parse arguments
import argparse

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net


    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)



class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, num_decoders=None):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.num_decoders = num_decoders

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        if isinstance(self.decoder, GaussianDecoderEnsemble):
            idx = np.random.choice(self.num_decoders, size=1)[0]
            elbo = torch.mean(
                self.decoder(z, idx).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
            )
        else:
            elbo = torch.mean(
                self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
            )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        if isinstance(self.decoder, GaussianDecoderEnsemble):
            idx = np.random.choice(self.num_decoders, size=1)[0]
            z = self.prior().sample(torch.Size([n_samples]))
            return self.decoder(z, idx).sample()
        else:
            z = self.prior().sample(torch.Size([n_samples]))
            return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)
    
class GaussianDecoderEnsemble(nn.Module):
    def __init__(self, decoder_nets):
        """
        Define a Gaussian decoder distribution based on a list of decoder networks.
        Samples a decoder randomly every forward batch.

        Parameters:
        decoder_net: list[torch.nn.Module]
           A list of decoder networks that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
           A single decoder should be sampled from the list
        """
        super(GaussianDecoderEnsemble, self).__init__()
        self.decoder_nets = nn.ModuleList(decoder_nets)

    def forward(self, z, idx=None):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        if idx is None:
            idx = np.random.choice(len(self.decoder_nets), size=1)[0]
            decoder_net = self.decoder_nets[idx]
        else:
            decoder_net = self.decoder_nets[idx]
        means = decoder_net(z)
        
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()

                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


class PLcurve:
    def __init__(self, x0, x1, N):
        super(PLcurve, self).__init__()
        self.x0 = x0.reshape(1,-1) # 1xD
        self.x1 = x1.reshape(1,-1) # 1xD
        self.N = N

        t = torch.linspace(0, 1, N).to(device).reshape(N,1) # Nx1
        c = (1-t) @ self.x0 + t @ self.x1 # NxD # Parametrisation of linear line
        self.params = c[1:-1] # (N-2)xD # Parameters between points are free to optimize, but in the ends not
        self.params.requires_grad = True

    def points(self):
        """Returns points of line"""
        c = torch.concatenate((self.x0, self.params, self.x1), axis=0) # NxD
        return c

    def plot(self):
        c = self.points().detach().cpu().numpy()
        plt.plot(c[:,0], c[:,1], color='k', alpha=0.6)

def curve_energy(model, curve, num_decoders=None):
    z = curve.points().to(device) 
    if isinstance(model.decoder, GaussianDecoderEnsemble):
        M = num_decoders if num_decoders is not None else model.num_decoders

        decoded_means = torch.stack(
            [model.decoder(z, idx=d).mean for d in range(M)],
            dim=0
        )  # (M, N, 1, 28, 28)

        dec1 = (decoded_means[:, 1:]).unsqueeze(1)
        dec2 = (decoded_means[:, :-1]).unsqueeze(0)

        delta = dec1 - dec2
        seg_energy = delta.pow(2).flatten(start_dim=3).sum(dim=3)  # Shape: (M, M, N-1)
        seg_energy = seg_energy.mean(dim=(0, 1)).sum()

        return seg_energy
               
    else:             
        mean_x = model.decoder(z).mean     # (N, 1, 28, 28)

        delta = mean_x[1:] - mean_x[:-1]   # consecutive decoded means
        segment_energy = delta.pow(2).flatten(start_dim=1).sum(dim=1)

    return segment_energy.sum()

def connecting_geodesic(model, curve, lr=1e-3, steps=2000, num_decoders=None):

    opt = optim.LBFGS([curve.params]
                      , lr=lr
                      , max_iter=steps
                      , line_search_fn='strong_wolfe')
    
    def closure():
        opt.zero_grad()
        energy = curve_energy(model, curve, num_decoders=num_decoders)
        energy.backward()
        return energy
        
    opt.zero_grad()
    opt.step(closure)


def encode_data_to_latent_space(model, mnist_data_loader):
    with torch.no_grad():

        latent_vars = []
        ys = []
        pos_means = []
        pos_stds = []

        data_iter = iter(mnist_data_loader)
        for batch in data_iter:
            x, y = batch[0].to(device), batch[1].to(device)

            # encode
            q = model.encoder(x)

            # get the dist 
            pos_mean, pos_std = q.mean, q.stddev
            pos_means.append(pos_mean)
            pos_stds.append(pos_std)

            # sample the latents
            latent = q.rsample()

            # save latents and label
            latent_vars.append(latent.reshape(-1, M))
            ys.append(y.reshape(-1, 1))

        latent_vars = torch.concatenate(latent_vars, axis=0)
        ys = torch.concatenate(ys, axis=0)
        pos_means = torch.concatenate(pos_means, axis=0)
        pos_stds = torch.concatenate(pos_stds, axis=0)
        return latent_vars, ys, pos_means, pos_stds

def plot_latent_space(latent_vars, ys, curve=None, save=False, plot_name = '.png'):
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(latent_vars[:, 0], latent_vars[:, 1], c=ys, alpha=0.6, cmap="tab10")
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    legend = plt.legend(handles, range(10), title="Class Label")
    if curve is not None:
        curve.plot()
    if save:
        plt.savefig("latent_space_" + plot_name)

def plot_latent_curves(model, latent_vars, num_curves, num_decoders=None):
    # sample 2*num_curves random points
    
    rd_points = np.random.choice(a=latent_vars.shape[0], size=num_curves*2,replace=False)
    rd_points = rd_points.reshape(2, num_curves)

    for i in tqdm(range(num_curves)):
        rd_idx_1, rd_idx_2 = rd_points[0, i], rd_points[1,i]
        x0 = torch.tensor(latent_vars[rd_idx_1,:]).to(device)
        x1 = torch.tensor(latent_vars[rd_idx_2,:]).to(device)

        c = PLcurve(x0, x1, 100)
        
        connecting_geodesic(model, c, num_decoders=num_decoders)
        c.plot() 

def plot_latent_pixel_uncertainty(model, latent_vars):
    n_grid_points = 100
    z1_max, z2_max = np.max(latent_vars, axis=0)
    z1_min, z2_min = np.min(latent_vars, axis=0)
    z1 = np.linspace(z1_min, z1_max, num=n_grid_points) # Nx1
    z2 = np.linspace(z2_min, z2_max, num=n_grid_points) # Nx1
    zz1, zz2 = np.meshgrid(z1,z2) # NxN, NxN

    zz1_tensor = torch.tensor(zz1.flatten()).reshape(-1,1) # (N^2)x1
    zz2_tensor = torch.tensor(zz2.flatten()).reshape(-1,1) # (N^2)x1
    zz = torch.concatenate([zz1_tensor, zz2_tensor], axis=1) # (N^2)x2
    if isinstance(model.decoder, GaussianDecoderEnsemble):
        means = []
        for idx in range(model.num_decoders):
            q = model.decoder(zz.to(device), idx)
            means.append(q.mean)
        means = torch.stack(means, dim=0) # num_decoders x (N^2) x 1 x 28 x 28
        mean_of_means = means.mean(dim=0) # (N^2) x 1 x 28 x 28
        stddev_means = torch.mean((means - mean_of_means)**2, dim=(0,2,3,4)) # (N^2)
        stddev_means_grid = stddev_means.reshape(n_grid_points, n_grid_points).detach().cpu().numpy()
    else:
        q = model.decoder(zz.to(device))
        stddev_means = torch.mean(q.stddev, dim=(1,2,3))
        stddev_means_grid = stddev_means.reshape(n_grid_points, n_grid_points).detach().cpu().numpy()
    heatmap = plt.contourf(zz1, zz2, stddev_means_grid, levels=100, cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Standard deviation of pixel values')

def new_encoder():
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )
    return encoder_net

def new_decoder():
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net


def get_VAE_model(num_decoders):
    if num_decoders > 1:
        decoder_nets = [new_decoder() for _ in range(num_decoders)]
        model = VAE(
            GaussianPrior(M),
            GaussianDecoderEnsemble(decoder_nets),
            GaussianEncoder(new_encoder()),
            num_decoders=num_decoders
        ).to(device)
    else:
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)

    return model

def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]

    return torch.utils.data.TensorDataset(new_data, new_targets)

def aggregate_cov_data(filename: str, dist_measure: str, num_decoders: int) -> pd.DataFrame:
    """
    Aggregates the raw distance results from a .npy file into the final cov results.

    Args:
    - filename:                 Name of .npy file.
    - dist_measure:             Distance measure, either 'euclidean' or 'geodesic'.
    - num_decoders:             Number of decoders. 
    
    Returns:
    - df_cov_per_num_decoders:  Dataframe containing average CoV across point pairs and VAEs.
    """
    # loading npy file
    raw_data = np.load(f"{filename}.npy", allow_pickle=True)

    # converting to pandas dataframes
    dist_data = raw_data.item()
    if dist_measure == "geodesic":
        df_dist = pd.DataFrame(
            [(k[0], k[1], k[2], k[3], v) for k, v in dist_data.items()],
            columns=["point1", "point2", "vae_idx", "num_decoders", "distance"]
            )
    elif dist_measure == "euclidean":
        df_dist = pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in dist_data.items()],
            columns=["point1", "point2", "vae_idx", "distance"]
        )
        # Repeat rows for all possible values of num_decoders
        df_dist = df_dist.loc[df_dist.index.repeat(num_decoders)].copy()
        df_dist["num_decoders"] = np.tile(np.arange(1, num_decoders + 1), len(df_dist) // num_decoders)
    else:
        ValueError("dist_measure argument must be either 'euclidean' or 'geodesic'")

    # computing mean (mu) and standard deviation (sigma) across VAEs for each number of decoders and point pair
    df_grouped = (
        df_dist
        .groupby(["point1", "point2", "num_decoders"])["distance"]
        .agg(
            mu="mean",
            sigma=lambda x: x.std(ddof=0)   # pandas uses sample standard deviation (divides by N-1 instead of N)
        ).reset_index()
    )

    # computing cov per point pair
    df_grouped["cov"] = df_grouped["sigma"] / df_grouped["mu"]

    # average cov across point pairs for each number of decoders
    df_cov_per_num_decoders = (
        df_grouped
        .groupby(["num_decoders"])["cov"]
        .agg(avg_cov="mean")
        .reset_index()
    )

    return df_cov_per_num_decoders

def load_data(num_train_data, num_classes):
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )
    return mnist_train_loader, mnist_test_loader

def load_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default='model',
        help="Model name without file extension (default: %(default)s)"
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    return parser

if __name__ == "__main__":
    parser = load_args()
    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device
    experiment_folder = args.experiment_folder
    M = args.latent_dim
    num_decoders = args.num_decoders
    epochs_per_decoder = args.epochs_per_decoder
    model_name = args.model_name

    num_train_data = 2048
    num_classes = 3
    mnist_train_loader, mnist_test_loader = load_data(num_train_data, num_classes)


    # Choose mode to run
    if args.mode == "train":
        
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        for rerun in range(args.num_reruns):
            model = get_VAE_model(args.num_decoders)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(
                model,
                optimizer,
                mnist_train_loader,
                args.epochs_per_decoder * args.num_decoders,
                args.device,
            )
            os.makedirs(f"{experiments_folder}", exist_ok=True)

            torch.save(
                model.state_dict(),
                f"{experiments_folder}/model_{rerun}.pt",
            )

    elif args.mode == "sample":
        model = get_VAE_model(args.num_decoders)
        model.load_state_dict(torch.load(args.experiment_folder + f"/model_{0}.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = get_VAE_model(args.num_decoders)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = get_VAE_model(args.num_decoders)
        model.load_state_dict(torch.load(args.experiment_folder + "/model_0.pt"))
        model.eval()

        if M > 2:
            raise NotImplementedError("Do not use more than two latent dimensions for this assignment")

        latent_vars, ys, _, _ = encode_data_to_latent_space(model, mnist_test_loader) # NxD, Nx1
        latent_vars, ys = latent_vars.cpu().numpy(), ys.cpu().numpy()

        plot_latent_space(latent_vars=latent_vars, ys=ys, save=False)
        plot_latent_pixel_uncertainty(model, latent_vars)
        plot_latent_curves(model, latent_vars, args.num_curves, num_decoders=args.num_decoders)
        
        plt.tight_layout()
        plt.savefig(f"{args.experiment_folder}/geodesics_{args.num_curves}_curves.png")

    elif args.mode == "cov":

        # select 10 random point pairs in the latent space
        print("Calculating geodesic distances and euclidean distances for random point pairs...")
        rd_points = np.random.choice(100, size=args.num_curves * 2, replace=True)
        rd_points = rd_points.reshape(2, args.num_curves)
        point_pairs_distances = {}
        point_pairs_distances_euclidian = {}
        for i in tqdm(range(args.num_curves)):
            print(f"Calculating distances for curve {i+1}/{args.num_curves}...")
            rd_idx_1, rd_idx_2 = rd_points[0, i], rd_points[1, i]

            print(f"Randomly selected point pair indices: {rd_idx_1}, {rd_idx_2}")
            for rerun in range(args.num_reruns):
                os.makedirs(f"{experiment_folder}", exist_ok=True)
                model = get_VAE_model()

                model.load_state_dict(torch.load(f"{experiment_folder}/{model_name}_{rerun}.pt"))

                model.eval()
                print("encoding data to latent space...")
                latent_vars, ys, _, _ = encode_data_to_latent_space(model, mnist_test_loader) # NxD, Nx1
                latent_vars, ys = latent_vars.cpu().numpy(), ys.cpu().numpy()

                for d in range(model.num_decoders):
                    # curve between latent
                    z1 = torch.tensor(latent_vars[rd_idx_1, :], dtype=torch.float32)
                    z2 = torch.tensor(latent_vars[rd_idx_2, :], dtype=torch.float32)
                    c = PLcurve(z1, z2, N=args.num_t, device=device, init_noise=0)
                    connecting_geodesic(model, c, lr=1e-4, steps=10000, num_decoders=d+1, mcmc_samples=30)
                    distance = c.distance().item()
                    point_pairs_distances[(rd_idx_1, rd_idx_2, rerun, d+1)] = distance   # d+1 to obtain number of decoders instead of index

                # Calculate Euclidean distances
                euclidean_distance = torch.norm(z1 - z2).item()
                point_pairs_distances_euclidian[(rd_idx_1, rd_idx_2, rerun)] = euclidean_distance   

        np.save("point_pairs_distances.npy", point_pairs_distances)
        np.save("point_pairs_distances_euclidean.npy", point_pairs_distances_euclidian)

        print("Finished calculating distances.")

        # aggregate raw CoV data from npy files 
        euclidean_covs = aggregate_cov_data("point_pairs_distances_euclidean", "euclidean", num_decoders)
        geodesic_covs = aggregate_cov_data("point_pairs_distances", "geodesic", num_decoders)

        # plotting CoVs in line chart
        plt.figure()
        plt.plot(euclidean_covs["num_decoders"], euclidean_covs["avg_cov"], label="Euclidean distance")
        plt.plot(geodesic_covs["num_decoders"], geodesic_covs["avg_cov"], label="Geodesic distance")
        plt.xticks(geodesic_covs["num_decoders"])
        plt.xlabel("Number of decoders")
        plt.ylabel("Coefficient of Variation")
        plt.legend()

        # saving plot as png
        filename = f"cov_plot_{num_decoders}_decoders.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved CoV plot as {filename}")