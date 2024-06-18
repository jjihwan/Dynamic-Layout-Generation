# Implement TransformerEncoder that can consider timesteps as optional args for Diffusion.

import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _gelu2(x):
    return x * F.sigmoid(1.702 * x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = True,
        norm_first: bool = True,
        device=None,
        dtype=None,
        # extension for diffusion
        diffusion_step: int = 100,
        timestep_type: str = 'adalayernorm',
    ) -> None:
        super().__init__()

        assert norm_first  # minGPT-based implementations are designed for prenorm only
        assert timestep_type in [
            None,
            "adalayernorm",
            "adainnorm",
            "adalayernorm_abs",
            "adainnorm_abs",
            "adalayernorm_mlp",
            "adainnorm_mlp",
        ]
        layer_norm_eps = 1e-5  # fixed

        self.norm_first = norm_first
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if timestep_type is not None:
            if "adalayernorm" in timestep_type:
                self.norm1 = AdaLayerNorm(d_model, diffusion_step, timestep_type)
            elif "adainnorm" in timestep_type:
                self.norm1 = AdaInsNorm(d_model, diffusion_step, timestep_type)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class TransformerEncoder(nn.Module):
    """
    Close to torch.nn.TransformerEncoder, but with timestep support for diffusion
    """

    __constants__ = ["norm"]

    def __init__(self, num_layers=4, dim_seq=10, dim_transformer=1024, nhead=8, dim_feedforward=2048,
                 diffusion_step=1000, device='cuda'):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = SinusoidalPosEmb(num_steps=25, dim=dim_transformer).to(device)
        pos_i = torch.tensor([i for i in range(25)]).to(device)
        self.pos_embed = self.pos_encoder(pos_i)

        self.layer_in = nn.Linear(in_features=dim_seq, out_features=dim_transformer).to(device)
        encoder_layer = Block(d_model=dim_transformer, nhead=nhead, dim_feedforward=dim_feedforward, diffusion_step=diffusion_step)
        self.layers = _get_clones(encoder_layer, num_layers).to(device)
        self.num_layers = num_layers
        self.layer_out = nn.Linear(in_features=dim_transformer, out_features=dim_seq).to(device)
    
    def from_pretrained(self, model_path):
        model_dict = torch.load(model_path)
        self.load_state_dict(model_dict)

        return self


    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        output = src

        output = self.layer_in(output)
        output = F.softplus(output)
        output = output + self.pos_embed

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
            ) # [B, L, D]; [2048, 25, 1024 or 512]

            if i < self.num_layers - 1:
                output = F.softplus(output)

        output = self.layer_out(output)

        return output


class TemporalTransformerEncoder(TransformerEncoder):
    def __init__(
            self,
            enable_temporal_layer: bool=True,
            is_train: bool=False,
            num_frame: int=4,
            pretrained_model_path: str=None,
            num_layers: int=4,
            dim_seq: int=10,
            dim_transformer: int=1024,
            nhead: int=8,
            dim_feedforward: int=2048,
            diffusion_step: int=1000,
            device: str='cuda',
    ):
        super().__init__(num_layers, dim_seq, dim_transformer, nhead, dim_feedforward, diffusion_step, device)
        # Load vanilla TransformerEncoder if train
        if pretrained_model_path is not None and is_train:
            self.from_pretrained(pretrained_model_path)

        self.enable_temporal_layer = enable_temporal_layer
        self.temporal_pos_encoder = SinusoidalPosEmb(num_steps=num_frame, dim=dim_transformer).to(device)
        pos_i = torch.tensor([i for i in range(num_frame)]).to(device)
        self.temporal_pos_embed = self.temporal_pos_encoder(pos_i)
        temporal_layer = Block(d_model=dim_transformer, nhead=nhead, dim_feedforward=dim_feedforward, diffusion_step=diffusion_step)
        self.temporal_layers = _get_clones(temporal_layer, num_layers).to(device)

        if pretrained_model_path is not None and not is_train:
            self.from_pretrained(pretrained_model_path)





    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        b, t, l, d = src.shape
        spatial_timestep = repeat(timestep, 'b -> (b t)', t=t).contiguous()
        temporal_timestep = repeat(timestep, 'b -> (b l)', l=l).contiguous()

        output = rearrange(src, 'b t l d -> (b t) l d').contiguous()

        output = self.layer_in(output)
        output = F.softplus(output)
        output = output + self.pos_embed

        for i, (layer, temporal_layer) in enumerate(zip(self.layers, self.temporal_layers)):
            output = layer(
                output,
                src_mask=None,
                src_key_padding_mask=None,
                timestep=spatial_timestep,
            )
            if self.enable_temporal_layer:
                output = rearrange(output, '(b t) l d -> (b l) t d', b=b)
                if i == 0:
                    output = output + self.temporal_pos_embed
                output = temporal_layer(
                    output,
                    src_mask=None,
                    src_key_padding_mask=None,
                    timestep=temporal_timestep,
                )
                output = rearrange(output, '(b l) t d -> (b t) l d', b=b)

            if i < self.num_layers - 1:
                output = F.softplus(output)

        output = self.layer_out(output)
        output = rearrange(output, '(b t) l d -> b t l d', b=b)

        return output
            



if __name__ == '__main__':
    dataset = "publaynet"
    if dataset == "publaynet":
        dim_transformer = 1024
        dim_seq = 10
    elif dataset == "rico25":
        dim_transformer = 512
        dim_seq = 30

    model = TemporalTransformerEncoder(
        enable_temporal_layer=True,
        pretrained_model_path=f"/home/kjh26720/code/LACE/model/{dataset}_best.pt",
        num_frame=4,
        num_layers=4,
        dim_seq=dim_seq,
        dim_transformer=dim_transformer,
        nhead=16,
        dim_feedforward=2048,
        diffusion_step=1000,
        device='cuda',
        )

    for name, param in model.named_parameters():
        if "temporal_layers" not in name:
            param.requires_grad_(False)

    input = torch.randn(512, 4, 25, 10).to('cuda')
    t = torch.randint(0, 1000, (512,)).to('cuda')
    output = model(input, timestep=t)
    print(output.mean(), output.std())
    breakpoint()










