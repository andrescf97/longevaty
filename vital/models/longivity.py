from vital.models.blocks import MAEVitEncoder, TransformerEncoder
from flax import nnx
import jax
import jax.numpy as jnp
from vital.models.rnn import RNN, Bidirectional
from vital.models.lungevity import CumProbLayer
from vital.models.attention import MultiHeadAttention


class BidirectionlBlock(nnx.Module):
    def __init__(self, cell, input_dim: int, hidden_dim: int, dtype: type = jnp.bfloat16, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.bidirectional = Bidirectional(
                    RNN(cell(input_dim, hidden_dim, dtype=dtype, rngs=rngs), unroll=3),
                    RNN(cell(input_dim, hidden_dim, dtype=dtype, rngs=rngs), unroll=3),
                )
        self.linear = nnx.Linear(hidden_dim * 2, hidden_dim, dtype=dtype, rngs=rngs)
    
    def __call__(self, inputs, masks, **kwargs):
        outputs = self.bidirectional(inputs, masks, **kwargs)
        outputs = self.linear(outputs)
        outputs = nnx.gelu(outputs)
        return outputs

class Longivity(nnx.Module):
    def __init__(
        self,
        pretrained_model_type: str = "pretrained",
        longitundinal_model: str = "rnn",
        rnn_cell: str = "simple",
        patch_size: int = 16,
        enc_hidden_dim: int = 768,
        rnn_hidden_dim: int = 384,
        hidden_dim: int = 512,
        max_followup: int = 6,
        enc_blocks: int = 12,
        enc_heads: int = 12,
        blocks: int = 5,
        heads: int = 12,
        bidirectional: bool = True,
        dropout_rate: float = 0.2,
        dtype: type = jnp.bfloat16,
        use_attention: bool = False,
        use_cls: bool = False,
        use_mean_token: bool = False,
        guided_attention_heads: int = 12,
        fusion_layer: bool = False,
        *,
        mlp_ratio: int = 4,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> nnx.Module:

        self.encoder = MAEVitEncoder(
            patch_size=patch_size,
            num_blocks=enc_blocks,
            num_heads=enc_heads,
            hidden_size=enc_hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.is_finetuned = False
        
        hidden = enc_hidden_dim
        if pretrained_model_type == "finetuned":
            self.mha = MultiHeadAttention(num_heads=guided_attention_heads, in_features=enc_hidden_dim, dtype=dtype, rngs=rngs,
                                        dropout_rate=dropout_rate, broadcast_dropout=False, decode=False, deterministic=True)

            # Default: use only attention pooling
            self.aggregate_fn = lambda x, y, z: x
            if use_cls:
                hidden += enc_hidden_dim
                self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, y], axis=-1)
            if use_mean_token:
                hidden += enc_hidden_dim
                self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, z], axis=-1)
            if use_cls and use_mean_token:
                self.aggregate_fn = lambda x, y, z: jnp.concatenate([x, y, z], axis=-1)

            self.is_finetuned = True

        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden), dtype=dtype))

        if longitundinal_model != "rnn":
            self.transformer = [
                TransformerEncoder(
                    hidden_size=hidden,
                    mlp_dim=hidden * mlp_ratio,
                    num_heads=heads,
                    dropout_rate=dropout_rate,
                    dtype=dtype,
                    rngs=rngs
                ) for _ in range(blocks)
            ]
            self.is_transformer = True
            self.final_norm = nnx.LayerNorm(hidden, rngs=rngs, dtype=dtype)

            self.time_embedding = nnx.Sequential(*[
                nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype),
                nnx.silu,
                nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype),
            ])
        else:
            self.init_layer, self.layers = make_rnn_layers(
                cell=rnn_cell,
                enc_hidden_dim=hidden,
                blocks=blocks,
                bidirectional=bidirectional,
                dtype=dtype,
                rngs=rngs
            )
            self.is_transformer = False

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
        return attns.squeeze(axis=1), attn_weights.mean(1).squeeze()


    def __call__(self, img0, img1, img2, t_mask, pos_embed, tim_embed):
        emb0 = self.encoder(img0, pos_embed)
        emb1 = self.encoder(img1, pos_embed)
        emb2 = self.encoder(img2, pos_embed)
        
        if self.is_finetuned:
            attn_pooled, attn_weights = self.attention_pooling(emb0)
            attn_pooled1, attn_weights1 = self.attention_pooling(emb1)
            attn_pooled2, attn_weights2 = self.attention_pooling(emb2)
            
            mean_pooled = emb0[:, 1:, :].mean(axis=1)
            mean_pooled1 = emb1[:, 1:, :].mean(axis=1)
            mean_pooled2 = emb2[:, 1:, :].mean(axis=1)
            
            cls = emb0[:, 0, :]
            cls1 = emb1[:, 0, :]
            cls2 = emb2[:, 0, :]

            pooled_output = self.aggregate_fn(attn_pooled, cls, mean_pooled)
            pooled_output1 = self.aggregate_fn(attn_pooled1, cls1, mean_pooled1)
            pooled_output2 = self.aggregate_fn(attn_pooled2, cls2, mean_pooled2)

            attn_weights_batch = jnp.stack((attn_weights, attn_weights1, attn_weights2), axis=1)
        else:
            pooled_output = emb0[:, 0, :]
            pooled_output1 = emb1[:, 0, :]
            pooled_output2 = emb2[:, 0, :]

            attn_weights_batch = None

        batch = jnp.stack((pooled_output, pooled_output1, pooled_output2), axis=1)

        if self.is_transformer:
            batch = self.time_embedding(tim_embed) + batch
            cls_token = jnp.tile(self.cls_token, [batch.shape[0], 1, 1])
            x = jnp.concatenate([cls_token, batch], axis=1) # TODO: check if this is needed
            x = self.dropout(x)
            t_mask = jnp.concatenate([jnp.ones((x.shape[0], 1)), t_mask], axis=1, dtype=jnp.bool)
            for layer in self.transformer:
                x = layer(x, mask=t_mask)
            x = self.final_norm(x)
            final_state = x[:, 0, :]

        else:
            output = self.init_layer(batch, t_mask)
            for layer in self.layers:
                output = layer(output, t_mask)

            final_state = output[:, -1, :] 

        op = self.classifier(final_state)
        return op, attn_weights_batch

def make_rnn_layers(
        cell: str = "simple",
        enc_hidden_dim: int = 768,
        blocks: int = 5,
        bidirectional: bool = True,
        dtype: type = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0)
):
    match cell:
        case "lstm":
            cell = nnx.nn.recurrent.LSTMCell
        case "gru":
            cell = nnx.nn.recurrent.GRUCell
        case "simple":
            cell = nnx.nn.recurrent.SimpleCell
        case _:
            cell = nnx.nn.recurrent.SimpleCell

    if bidirectional:
        init_layer = BidirectionlBlock(cell=cell, input_dim=enc_hidden_dim, hidden_dim=enc_hidden_dim, dtype=dtype, rngs=rngs)
        layers = [
            BidirectionlBlock(cell=cell, input_dim=enc_hidden_dim, hidden_dim=enc_hidden_dim, dtype=dtype, rngs=rngs)
            for _ in range(blocks - 1)
        ]
    else:
        init_layer = RNN(cell(enc_hidden_dim, enc_hidden_dim, dtype=dtype, rngs=rngs), unroll=3)
       
        layers = [
            RNN(cell(enc_hidden_dim, enc_hidden_dim, dtype=dtype, rngs=rngs), unroll=3) 
            for _ in range(blocks - 1)
        ]
    return init_layer, layers


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