from vital.models.blocks import MAEVitEncoder
from flax import nnx
import jax.numpy as jnp
from vital.models.rnn import RNN
from vital.models.lungevity import CumProbLayer

class Longivity(nnx.Module):
    def __init__(
        self,
        patch_size: int = 16,
        enc_hidden_dim: int = 768,
        rnn_hidden_dim: int = 384,
        hidden_dim: int = 512,
        max_followup: int = 6,
        blocks: int = 12,
        heads: int = 12,
        dropout_rate: float = 0.2,
        dtype: type = jnp.bfloat16,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> nnx.Module:

        self.encoder = MAEVitEncoder(
            patch_size=patch_size,
            num_blocks=blocks,
            num_heads=heads,
            hidden_size=enc_hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        cell = nnx.nn.recurrent.SimpleCell(enc_hidden_dim, rnn_hidden_dim,
                                                dtype=dtype,
                                                rngs=rngs)
        self.rnn = RNN(cell, return_carry=True)
        self.classifier = nnx.Sequential(*[
            nnx.Linear(rnn_hidden_dim, hidden_dim, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            CumProbLayer(hidden_dim, max_followup, rngs=rngs, dtype=dtype)
        ])

    def __call__(self, img0, img1, img2, t_mask, pos_embed):
        emb0 = self.encoder(img0, pos_embed)[:, 0, :]
        emb1 = self.encoder(img1, pos_embed)[:, 0, :]
        emb2 = self.encoder(img2, pos_embed)[:, 0, :]

        batch = jnp.stack((emb0, emb1, emb2), axis=1)
        h, _ = self.rnn(batch, t_mask)
        op = self.classifier(h)
        return op


