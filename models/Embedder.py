import torch
import torch.nn as nn


# Positional encoding embedding. Code is a modified version from https://github.com/bmild/nerf.
class Embedder(nn.Module):
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 10,
        num_freqs: int = 10,
        log_sampling: bool = True,
    ):
        super(Embedder, self).__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        else:
            self.freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, num_freqs)

        self.out_dim = 0
        if include_input:
            self.out_dim += input_dims
        self.out_dim += 2 * input_dims * len(self.freq_bands)  # for sin and cos

    def forward(self, inputs):
        embeddings = [inputs] if self.include_input else []
        for freq in self.freq_bands:
            for p_fn in (torch.sin, torch.cos):
                embeddings.append(p_fn(inputs * freq))
        return torch.cat(embeddings, -1)


def get_embedder(multires, input_dims=3):
    # Move the embedder to the appropriate device and set it to evaluation mode
    embedder_obj = Embedder(
        input_dims=input_dims,
        max_freq_log2=multires - 1,
        num_freqs=multires,
    )
    embedder_obj.eval()  # Optionally set the module to evaluation mode
    return embedder_obj, embedder_obj.out_dim
