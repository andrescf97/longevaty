from vital.models.blocks import MAEVitEncoder, TransformerEncoder
from flax import nnx
import jax.numpy as jnp
from vital.models.rnn import RNN, Bidirectional
from vital.models.lungevity import CumProbLayer

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
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, enc_hidden_dim), dtype=dtype))

        if longitundinal_model != "rnn":
            self.transformer = [
                TransformerEncoder(
                    hidden_size=enc_hidden_dim,
                    mlp_dim=enc_hidden_dim * mlp_ratio,
                    num_heads=heads,
                    dropout_rate=dropout_rate,
                    dtype=dtype,
                    rngs=rngs
                ) for _ in range(blocks)
            ]
            self.is_transformer = True
            self.final_norm = nnx.LayerNorm(enc_hidden_dim, rngs=rngs, dtype=dtype)

            self.time_embedding = nnx.Sequential(*[
                nnx.Linear(enc_hidden_dim, enc_hidden_dim, rngs=rngs, dtype=dtype),
                nnx.silu,
                nnx.Linear(enc_hidden_dim, enc_hidden_dim, rngs=rngs, dtype=dtype),
            ])
        else:
            self.init_layer, self.layers = make_rnn_layers(
                cell=rnn_cell,
                enc_hidden_dim=enc_hidden_dim,
                rnn_hidden_dim=rnn_hidden_dim,
                blocks=blocks,
                bidirectional=bidirectional,
                dtype=dtype,
                rngs=rngs
            )
            self.is_transformer = False

        self.classifier = nnx.Sequential(*[
            nnx.Linear(rnn_hidden_dim, hidden_dim, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            CumProbLayer(hidden_dim, max_followup, rngs=rngs, dtype=dtype)
        ])


    def __call__(self, img0, img1, img2, t_mask, pos_embed, tim_embed):
        emb0 = self.encoder(img0, pos_embed)[:, 0, :]
        emb1 = self.encoder(img1, pos_embed)[:, 0, :]
        emb2 = self.encoder(img2, pos_embed)[:, 0, :]

        batch = jnp.stack((emb0, emb1, emb2), axis=1)

        if self.is_transformer:
            batch = self.time_embedding(tim_embed) + batch
            cls_token = jnp.tile(self.cls_token, [batch.shape[0], 1, 1])
            x = jnp.concatenate([cls_token, batch], axis=1)
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
        return op

def make_rnn_layers(
        cell: str = "simple",
        enc_hidden_dim: int = 768,
        rnn_hidden_dim: int = 384,
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
        init_layer = BidirectionlBlock(cell=cell, input_dim=enc_hidden_dim, hidden_dim=rnn_hidden_dim, dtype=dtype, rngs=rngs)
        layers = [
            BidirectionlBlock(cell=cell, input_dim=rnn_hidden_dim, hidden_dim=rnn_hidden_dim, dtype=dtype, rngs=rngs)
            for _ in range(blocks - 1)
        ]
    else:
        init_layer = RNN(cell(enc_hidden_dim, rnn_hidden_dim, dtype=dtype, rngs=rngs), unroll=3)
       
        layers = [
            RNN(cell(rnn_hidden_dim, rnn_hidden_dim, dtype=dtype, rngs=rngs), unroll=3) 
            for _ in range(blocks - 1)
        ]
    return init_layer, layers