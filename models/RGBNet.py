import torch
import torch.nn as nn
from models.Embedder import get_embedder


class RGBNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        d_feats: int,
        n_hidden: int,
        multires: int = 0,
    ):
        super(RGBNet, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_feats = d_feats
        self.d_hidden = d_hidden
        self.multires = multires
        self.layers = nn.ModuleList([])
        self.embedder = None

        if multires > 0:
            self.embedder, d_in_embd = get_embedder(multires)
            self.d_in = d_in_embd + d_in - 3

        self.dims = (
            [self.d_in + self.d_feats]
            + [self.d_hidden for _ in range(n_hidden)]
            + [self.d_out]
        )

        self.num_layers = len(self.dims)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for i in range(self.num_layers - 1):
            L = nn.Linear(self.dims[i], self.dims[i + 1])
            L = self.normal_layer(L)
            self.layers.append(L)

    def normal_layer(self, layer):
        return nn.utils.parametrizations.weight_norm(layer)

    def forward(self, points, normals, view_dirs, feats):
        if self.embedder is not None:
            view_dirs = self.embedder(view_dirs)

        X = torch.cat([points, view_dirs, normals, feats], dim=-1)

        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i < self.num_layers - 2:
                X = self.activation(X)

        X = self.sigmoid(X)

        return X
