import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx

class PatchEmbed1D(nnx.Module):
    def __init__(self, patch_size=16, embed_dim=768, patch_dim=3, dtype=jnp.bfloat16, rngs = nnx.Rngs(0)):
        input_features = pow(patch_size, patch_dim)
        self.proj = nnx.Linear(input_features, embed_dim, rngs=rngs, dtype=dtype)
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(x)

class MAEVitEncoder(nnx.Module):
    def __init__(
        self,
        patch_size: int = 16,
        num_blocks: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        *,
        dtype: type = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # Patch and position embedding
        self.patch_embeddings = PatchEmbed1D(patch_size=patch_size, embed_dim=hidden_size, rngs=rngs, dtype=dtype)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size), dtype=dtype))

        # Transformer Encoder blocks
        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, hidden_size * mlp_ratio, num_heads, dropout_rate, rngs=rngs, dtype=dtype)
            for i in range(num_blocks)
        ])
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs, dtype=dtype)


    def __call__(self, x: jax.Array, pos_embed: jax.Array) -> jax.Array:
        # Patch and position embedding
        patches = self.patch_embeddings(x)
        batch_size = patches.shape[0]

        cls_token = jnp.tile(self.cls_token, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + pos_embed
        embeddings = self.dropout(embeddings)

        # Encoder blocks
        x = self.encoder(embeddings)
        x = self.final_norm(x)
        return x

class MAEVitDecoder(nnx.Module):
    def __init__(
        self,
        patch_size: int = 16,
        patch_dim: int = 3,
        embed_dim: int = 768,
        num_blocks: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout_rate: float = 0.1,
        *,
        dtype: type = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.out_features = pow(patch_size, patch_dim)
        self.embed_dim = embed_dim
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # Transformer Encoder blocks
        self.decoder = nnx.Sequential(*[
            TransformerEncoder(embed_dim, embed_dim * mlp_ratio, num_heads, dropout_rate, rngs=rngs, dtype=dtype)
            for i in range(num_blocks)
        ])
        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs, dtype=dtype)
        self.projector = nnx.Linear(embed_dim, self.out_features, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array, pos_embed: jax.Array) -> jax.Array:
        input = x + pos_embed
        x = self.dropout(input)
        x = self.decoder(x)
        x = self.final_norm(x)
        x = self.projector(x)
        return x


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        dtype: type = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:

        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs, dtype=dtype)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
            dtype=dtype
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs, dtype=dtype)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs, dtype=dtype),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def build_3d_sincos_position_embedding(batch, grid_size, embed_dim, temperature=10000., dtype=jnp.bfloat16):
    # Ensure grid size is in the correct format
    h, w, d = grid_size
    # Create 1D grids for h, w, and d
    grid_h = jnp.arange(h, dtype=dtype)
    grid_w = jnp.arange(w, dtype=dtype)
    grid_d = jnp.arange(d, dtype=dtype)
    
    # Create 3D meshgrid
    grid_h, grid_w, grid_d = jnp.meshgrid(grid_h, grid_w, grid_d, indexing='ij')

    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'

    pos_dim = embed_dim // 6
    omega = jnp.arange(pos_dim, dtype=dtype) / pos_dim
    omega = 1. / (temperature ** omega)

    # Flatten grids and apply omega scaling
    out_h = jnp.einsum('m,d->md', grid_h.flatten(), omega)
    out_w = jnp.einsum('m,d->md', grid_w.flatten(), omega)
    out_d = jnp.einsum('m,d->md', grid_d.flatten(), omega)

    # Compute sin and cos embeddings
    pos_emb = jnp.concatenate([
        jnp.sin(out_h), jnp.cos(out_h),
        jnp.sin(out_w), jnp.cos(out_w),
        jnp.sin(out_d), jnp.cos(out_d)
    ], axis=1)

    # Reshape the embeddings to (1, num_positions, embed_dim)
    pos_emb = jnp.concatenate([jnp.zeros((1, pos_emb.shape[1])), pos_emb], axis=0)
    pos_emb = jnp.tile(pos_emb, (batch, 1, 1))
    return pos_emb

def build_1d_sincos_position_embedding(
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    temperature: float = 10000.,
    dtype=jnp.bfloat16,
) -> jax.Array:
    """
    Builds a 1D sine-cosine position embedding for a sequence.

    Args:
        batch_size: The batch size.
        sequence_length: The length of the sequence (number of tokens).
        embed_dim: The dimension of the embeddings.
        temperature: The temperature parameter for the sine-cosine calculation.
        dtype: The data type for the embeddings.
        add_cls_token_embedding: If True, prepends a row of zeros to the position
                                 embeddings, suitable for a CLS token.

    Returns:
        A JAX array of shape (batch_size, num_positions, embed_dim),
        where num_positions is sequence_length (+1 if add_cls_token_embedding is True).
    """

    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'

    # Create 1D grid for the sequence length
    grid_l = jnp.arange(sequence_length, dtype=dtype)

    pos_dim = embed_dim // 2 # Half for sin, half for cos
    omega = jnp.arange(pos_dim, dtype=dtype) / pos_dim
    omega = 1. / (temperature ** omega)

    # Apply omega scaling
    # jnp.einsum('m,d->md', grid_l, omega) calculates outer product
    out_l = jnp.einsum('m,d->md', grid_l, omega)

    # Compute sin and cos embeddings
    pos_emb = jnp.concatenate([
        jnp.sin(out_l), jnp.cos(out_l)
    ], axis=1) # Concatenate along the embedding dimension

    pos_emb = jnp.concatenate([jnp.zeros((1, embed_dim), dtype=dtype), pos_emb], axis=0)
    pos_emb = jnp.tile(pos_emb, (batch_size, 1, 1))

    return pos_emb