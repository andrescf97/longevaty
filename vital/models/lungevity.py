from vital.models.blocks import MAEVitEncoder
from flax import nnx
import jax
import jax.numpy as jnp
from vital.models.attention import MultiHeadAttention

class CumProbLayer(nnx.Module):
    def __init__(
            self,
            num_features: int = 768,
            max_followup: int = 6,
            dtype: type = jnp.bfloat16,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.hazard_layer = nnx.Linear(num_features, max_followup, dtype=dtype, rngs=rngs)
        self.base_hazard_layer = nnx.Linear(num_features, 1, dtype=dtype, rngs=rngs)

        mask = jnp.ones((max_followup, max_followup))
        mask = jnp.tril(mask, k=0).T
        self.mask = nnx.Variable(mask)

    def __call__(self, x: jax.Array):
        base_hazard = self.base_hazard_layer(x)

        hazards = self.hazard_layer(x)
        hazards = nnx.relu(hazards)
        expanded_hazards = jnp.tile(hazards[:, :, None], (1, 1, hazards.shape[-1]) ) # (B, T, T) from (B, T)
        masked_hazards = expanded_hazards * self.mask

        return masked_hazards.sum(axis=1) + base_hazard

class ResidualBlock(nnx.Module):
    def __init__(
            self,
            dim: int = 1024,
            hidden_dim: int = 768,
            dropout_rate: float = 0.2,
            dtype: type = jnp.bfloat16,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.fc1 = nnx.Linear(dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(rate = dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, dtype=dtype, rngs=rngs)

    def __call__( self, x):
        identity = x
        out = nnx.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out + identity


class FusionLayerWithResidual(nnx.Module):
    def __init__(
            self,
            input_dim: int = 2304,
            hidden_dim: int = 1024,
            output_dim: int = 768,
            dropout_rate: float = 0.1,
            num_residual_blocks: int = 2,
            dtype: type = jnp.bfloat16,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.fc_in = nnx.Linear(input_dim, input_dim, dtype=dtype, rngs=rngs)
        self.res_blocks = nnx.Sequential(*[
            ResidualBlock(input_dim, hidden_dim, dropout_rate, dtype=dtype,rngs=rngs)
            for _ in range(num_residual_blocks)
        ])
        self.fc_out = nnx.Linear(input_dim, output_dim, dtype=dtype, rngs=rngs)

    def __call__(self, x):
        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = nnx.gelu(self.fc_out(x))
        return x

class LungeVity(nnx.Module):
    def __init__(
        self, 
        patch_size: int = 16,
        hidden_dim: int = 768,
        max_followup: int = 6,
        blocks: int = 12,
        heads: int = 12,
        dropout_rate: float = 0.2,
        dtype: type = jnp.bfloat16,
        use_cls: bool = False,
        use_mean_token: bool = False,
        guided_attention_heads: int = 8,
        fusion_layer: bool = False,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> nnx.Module:

        self.encoder = MAEVitEncoder(
            patch_size=patch_size,
            num_blocks=blocks,
            num_heads=heads,
            hidden_size=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        hidden = hidden_dim
        self.aggregate_fn = lambda x, y, z: x
        if use_cls:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, y], axis=-1)
        if use_mean_token:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, z], axis=-1)
        if use_cls and use_mean_token:
            self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, y, z], axis=-1)

        self.mha = MultiHeadAttention(num_heads=guided_attention_heads, in_features=hidden_dim, dtype=dtype, rngs=rngs,
                                      dropout_rate=dropout_rate, broadcast_dropout=False, decode=False, deterministic=True)

        if fusion_layer:
            self.classifier = nnx.Sequential(*[
                FusionLayerWithResidual(input_dim=hidden, output_dim=hidden_dim, hidden_dim=1024, dropout_rate=dropout_rate, dtype=dtype, rngs=rngs),
                CumProbLayer(hidden_dim, max_followup, rngs=rngs, dtype=dtype)
            ])
        else:
            self.classifier = nnx.Sequential(*[
                nnx.Linear(hidden, hidden_dim, rngs=rngs, dtype=dtype),
                nnx.gelu,
                nnx.Dropout(rate=dropout_rate, rngs=rngs),
                CumProbLayer(hidden_dim, max_followup, rngs=rngs, dtype=dtype)
            ])

    def attention_pooling(
            self,
            tokens: jax.Array
    ):
        q = tokens[:, 0:1, :]
        k = tokens[:, 1:, :]
        v = tokens[:, 1:, :]

        attns, attn_weights = self.mha(q, k, v)
        return attns.squeeze(), attn_weights.mean(1).squeeze()

    def __call__(
        self,
        input: jax.Array,
        pos_embed: jax.Array
    ):
        embeddings = self.encoder(input, pos_embed)

        attn_pooled, attn_weights = self.attention_pooling(embeddings)
        mean_pooled = embeddings[:, 1:, :].mean(axis=1)
        cls = embeddings[:, 0, :]
        
        pooled_output = self.aggregate_fn(attn_pooled, cls, mean_pooled)
        op = self.classifier(pooled_output)
        return op, attn_weights

