import numpy as np
import torch as th
from src.utils.data import load_data, estimate_token_frequency


def create_mask(
    x: th.Tensor, 
    token_frequencies: th.Tensor = None, 
    n_masked: int = 4,
    rand_proportion: float = 0.1,
):
    if token_frequencies is None:
        token_frequencies = np.ones(x.shape[1])
    token_counts_per_frame = x.sum(dim=(2,3))
    active_tokens_per_frame = token_counts_per_frame > 0
    token_probs = active_tokens_per_frame / token_frequencies.reshape(1,-1)
    token_probs /= token_probs.sum(1, keepdim=True)

    token_probs += rand_proportion * active_tokens_per_frame
    token_probs /= token_probs.sum(1, keepdim=True)

    samples = th.distributions.Categorical(probs=token_probs).sample()

    token_filter = th.zeros_like(x)
    token_filter[np.arange(x.shape[0]), samples, :, :] = 1

    counts_per_frame = token_counts_per_frame[np.arange(x.shape[0]), samples]
    tile_number_filter = (
        th.rand_like(x) 
        < (n_masked / counts_per_frame.reshape(-1,1,1,1))   # Should be larger than 0 by design
    )

    mask = 1 - token_filter * tile_number_filter
    x_masked = x * mask
    return x_masked, mask


if __name__ == "__main__":

    train_percentage = 0.8
    train_tensor, val_tensor = load_data(train_percentage)
    total_tensor = th.cat([train_tensor, val_tensor], axis=0)
    frequencies = estimate_token_frequency(total_tensor)
    print(frequencies)

    x_masked, mask = create_mask(total_tensor, frequencies)
    difs = (total_tensor - x_masked).sum((1,2,3))
    print(difs.mean())