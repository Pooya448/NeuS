import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Embedder import get_embedder


class SDFNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_hidden: int,
        skips: list = [4],
        multires: int = 0,
        bias: float = 0.5,
        scale: float = 1.0,
    ):
        super(SDFNet, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.multires = multires
        self.skips = skips
        self.bias = bias
        self.layers = nn.ModuleList([])
        self.embedder = None
        self.scale = scale

        if self.multires > 0:
            self.embedder, d_in_embd = get_embedder(self.multires, input_dims=d_in)
            self.d_in = d_in_embd

        self.dims = (
            [self.d_in] + [self.d_hidden for _ in range(n_hidden)] + [self.d_out]
        )
        self.num_layers = len(self.dims)
        self.activation = nn.Softplus(beta=100)

        for i in range(self.num_layers - 1):
            out_dim = (
                (self.dims[i + 1] - self.d_in)
                if i + 1 in self.skips
                else self.dims[i + 1]
            )

            L = nn.Linear(self.dims[i], out_dim)

            L = self.geo_init(L, i, out_dim)
            L = self.normal_layer(L)

            self.layers.append(L)

    def forward(self, inputs):

        inputs = inputs * self.scale
        if self.embedder is not None:
            inputs = self.embedder(inputs)

        X = inputs

        for i, layer in enumerate(self.layers):

            if i in self.skips:
                X = torch.cat([X, inputs], dim=-1) / np.sqrt(2)  #! CHECK

            X = layer(X)

            if i < self.num_layers - 2:
                X = self.activation(X)

        sdf = X[:, :1] / self.scale
        feats = X[:, 1:]

        return sdf, feats

    def sdf(self, X):
        sdf, _ = self.forward(X)
        return sdf

    def gradient(self, X):
        X = X.requires_grad_(True)
        y, _ = self.forward(X)
        out = torch.ones_like(y, requires_grad=False, device=y.device)
        grads = torch.autograd.grad(
            outputs=y,
            inputs=X,
            grad_outputs=out,
            create_graph=True,
            retain_graph=True,
        )[0]
        return grads

    def normal_layer(self, layer):
        return nn.utils.parametrizations.weight_norm(layer, dim=0)

    def geo_init(self, layer, idx, layer_out_dim):
        if idx == self.num_layers - 2:
            nn.init.normal_(
                layer.weight, mean=np.sqrt(np.pi) / np.sqrt(self.dims[idx]), std=0.0001
            )
            nn.init.constant_(layer.bias, self.bias * -1)

        elif self.multires > 0 and idx == 0:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.constant_(layer.weight[:, 3:], 0.0)
            nn.init.normal_(
                layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(layer_out_dim)
            )

        elif self.multires > 0 and idx in self.skips:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer_out_dim))
            nn.init.constant_(layer.weight[:, -(self.dims[0] - 3) :], 0.0)

        else:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer_out_dim))

        return layer
