from vital.models.blocks import MAEVitDecoder, MAEVitEncoder
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

class Vital(nnx.Module):
    def __init__(
        self,
        patch_size: int = 16,
        enc_dim: int = 768,
        dec_dim: int = 768,
        dec_blocks: int = 12,
        dec_heads: int = 12,
        drouput_rate: float = 0.2,
        dtype: type = jnp.bfloat16,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.encoder = MAEVitEncoder(
            patch_size=patch_size,
            num_blocks=12,
            num_heads=12,
            hidden_size=enc_dim,
            dropout_rate=drouput_rate,
            dtype=dtype,
            rngs=rngs
        )
        self.decoder = MAEVitDecoder(
            patch_size=patch_size,
            embed_dim=dec_dim,
            num_blocks=dec_blocks,
            num_heads=dec_heads,
            dropout_rate=drouput_rate,
            dtype=dtype,
            rngs=rngs
        )
        self.encoding_projection = nnx.Linear(enc_dim, dec_dim, rngs=rngs, dtype=dtype)

        self.mask_token = nnx.Param(jnp.zeros((1, 1, dec_dim), dtype=dtype))

    def __call__(
        self,
        input: jax.Array,
        enc_pos_embed: jax.Array,
        dec_pos_embed: jax.Array,
        selected_indices: jax.Array,
        masked_indices: jax.Array,
    ) -> jax.Array:
        B, _, _ = input.shape
        embeddings = self.encoder(input, enc_pos_embed)
        projected_embeddings = self.encoding_projection(embeddings)

        masked_tokens = jnp.tile(self.mask_token, (B, masked_indices.shape[1], 1))
        all_embeddings = jnp.concatenate([projected_embeddings, masked_tokens], axis=1)
        all_indices = jnp.concatenate([selected_indices, masked_indices], axis=1)

        shuffled_pos_embed = np.take_along_axis(dec_pos_embed, all_indices, axis=1)
        shuffled_recon_image = self.decoder(all_embeddings, shuffled_pos_embed)
        return shuffled_recon_image[:, 1:, :]

