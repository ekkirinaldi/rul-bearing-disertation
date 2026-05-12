import numpy as np
import torch

from mxlstm.interp.sae import SAEConfig, TopKSparseAutoencoder, train_sae


def test_sae_forward_shape():
    sae = TopKSparseAutoencoder(SAEConfig(d_model=16, expansion=4, k=8))
    x = torch.randn(32, 16)
    x_hat, z = sae(x)
    assert x_hat.shape == x.shape
    assert z.shape == (32, 16 * 4)
    nonzero_per_row = (z != 0).sum(dim=-1)
    assert int(nonzero_per_row.max()) <= 8


def test_sae_train_decreases_recon():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((512, 16)).astype(np.float32)
    sae = TopKSparseAutoencoder(SAEConfig(d_model=16, expansion=4, k=4))
    history = train_sae(sae, h, epochs=5, batch_size=64, lr=1e-2)
    assert history[-1]["recon"] < history[0]["recon"]
