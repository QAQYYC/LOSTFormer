import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from layers.Revin import RevIN
from layers.fast_attention_spatio_cayley import PerformerAttention
from layers.SparseMoE import SparseMoE
from copy import deepcopy


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TemporalEmbedding(nn.Module):
    def __init__(self, config, d_model, seq_len, num_patches, channels, dropout=0.1):
        super(TemporalEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embeddings = nn.Linear(channels, d_model, bias=False)
        self.num_patches = num_patches
        self.channels = channels
        self.d_model = d_model
        self.d_ff = config.d_ff
        self.transformer_block = TransformerBlock(
            config=config,
            masked_attention=PerformerAttention(config, nb_features=None, use_relu_kernel=False, d_model=d_model),
            self_attention=PerformerAttention(config, nb_features=None, use_relu_kernel=False, d_model=d_model),
            feed_forward=SparseMoE(d_model, self.d_ff, num_experts=4, top_k=2),
            d_model=d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        Args:
            x: [batch, seq_len, channels]

        Returns:
            x: [batch, 1, num_patches, d_model]

        """
        x = x.reshape(x.shape[:-2] + (self.channels,))
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x, _ = self.transformer_block(x)

        return self.dropout(x)


class SpatialEmbedding(nn.Module):
    def __init__(self, config, d_model, seq_len, num_patches, channels, dropout=0.1):
        super(SpatialEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embeddings = nn.Linear(channels, d_model, bias=False)
        self.num_patches = num_patches
        self.channels = channels
        self.d_model = d_model
        self.d_ff = config.d_ff
        self.transformer_block = TransformerBlock(
            config=config,
            masked_attention=PerformerAttention(config, nb_features=None, use_relu_kernel=False, d_model=d_model),
            self_attention=PerformerAttention(config, nb_features=None, use_relu_kernel=False, d_model=d_model),
            feed_forward=SparseMoE(d_model, self.d_ff, num_experts=4, top_k=2),
            d_model=d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        Args:
            x: [batch, seq_len, channels]

        Returns:
            x: [batch, num_nodes, 1, d_model]

        """
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[:-2] + (self.channels,))
        x = self.embeddings(x)
        x = x.unsqueeze(2)
        x, _ = self.transformer_block(x)
        return self.dropout(x)


class MultiHeadsDotAttention(nn.Module):
    def __init__(self, config, d_v=None):
        super(MultiHeadsDotAttention, self).__init__()
        self.device = config.device
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.key_projection = nn.Linear(config.d_model, config.d_model)
        self.value_projection = nn.Linear(config.d_model, config.d_model)

        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_model = config.d_model

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads
        if d_v is None:
            self.d_v = self.d_model // self.n_heads

    def forward(self, query, key, value, masked=False):
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        q_shape = query.shape
        k_shape = key.shape
        P, D = query.shape[-2:]
        K_P, D = key.shape[-2:]
        query = query.reshape(q_shape[:-2] + (P, self.n_heads, self.d_k)).transpose(-2, -3)
        key = key.reshape(q_shape[:-2] + (K_P, self.n_heads, self.d_k)).transpose(-2, -3)
        value = value.reshape(q_shape[:-2] + (K_P, self.n_heads, self.d_v)).transpose(-2, -3)

        attn_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if masked is True:
            masked_mat = torch.tril(torch.ones([P, P]), diagonal=0).to(torch.bool).to(self.device)
            attn_score = torch.masked_fill(attn_score, ~masked_mat, -np.inf)

        output = torch.softmax(attn_score, dim=-1)
        output = output @ value
        output = rearrange(output, '... h p d -> ... p (h d)')
        output = self.output_projection(output)

        return output, attn_score


class TransformerBlock(nn.Module):
    def __init__(self, config, masked_attention, self_attention, feed_forward=None, d_model=None):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model if d_model is not None else config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        self.masked_attention = masked_attention
        self.self_attention = self_attention

        if feed_forward is None:
            self.feed_forward = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.ReLU(),
                nn.Linear(self.d_ff, self.d_model),
            )
        else:
            self.feed_forward = feed_forward

    def forward(self, x):
        residual = x
        x, attn_score = self.self_attention(x, x, x)
        x = self.norm2(residual + self.dropout2(x))

        residual = x
        x = self.feed_forward(x)
        x = self.norm3(residual + self.dropout3(x))
        return x, attn_score


class SeriesDecomposition(nn.Module):
    """
    Series decomposition module that separates time series into trend and seasonal components
    """

    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x shape: [batch, seq_len, channels] or [batch, seq_len, num_nodes, channel]
        original_shape = x.shape

        # Reshape for pooling
        if len(original_shape) == 4:
            # [batch, seq_len, num_nodes, channel] -> [batch, num_nodes * channel, seq_len]
            x_reshaped = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        else:
            # [batch, seq_len, channels] -> [batch, channels, seq_len]
            x_reshaped = x.transpose(1, 2)

        # Apply moving average to extract trend component
        trend = self.avg_pool(x_reshaped)

        # Seasonal component is the residual
        seasonal = x_reshaped - trend

        # Reshape back to original shape
        if len(original_shape) == 4:
            trend = trend.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            seasonal = seasonal.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
        else:
            trend = trend.transpose(1, 2)
            seasonal = seasonal.transpose(1, 2)

        return trend, seasonal


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config.device
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.enc_in = config.enc_in
        self.num_nodes = config.num_nodes
        self.channel = config.channel
        self.d_ff = config.d_ff
        self.slice_size_per_day = config.slice_size_per_day
        self.adj_mx = None

        self.revin = RevIN(config.enc_in, affine=False)

        self.embedding = nn.Linear(self.channel + 2, self.d_model)
        self.series_decomposition = SeriesDecomposition(kernel_size=3)
        self.tod_embedding = nn.Embedding(self.slice_size_per_day, self.d_model)
        self.dow_embedding = nn.Embedding(7, self.d_model)

        self.spatial_embedding = SpatialEmbedding(
            config,
            self.d_model,
            self.seq_len,
            self.seq_len,
            config.seq_len * config.channel
        )
        self.adaptive_embeddings = nn.Embedding(self.seq_len * self.num_nodes,
                                                self.d_model).to(self.device)

        self.emb_projection = nn.Linear(self.d_model * 6, self.d_model)

        self.transformers_list = nn.ModuleList([
            TransformerBlock(
                config,
                PerformerAttention(
                    config,
                    nb_features=None,
                    use_relu_kernel=False
                ),
                PerformerAttention(
                    config,
                    nb_features=None,
                    use_relu_kernel=False
                ),
                feed_forward=SparseMoE(self.d_model, self.d_ff, num_experts=4, top_k=2)
            ) for i in range(config.d_layers)
        ])

        self.up_head = nn.Linear(self.d_model, self.channel)

        self.mix_projection = nn.Linear(self.seq_len * self.d_model, self.pred_len * self.channel)

    def forecast(self, x, x_mark=None):
        need_transpose = False
        if len(x.shape) == 3:
            need_transpose = True
            x = x.unsqueeze(-1)

        x = self.revin(x, 'norm')

        x_trend, x_season = self.series_decomposition(x)

        spatio_emb = self.spatial_embedding(x)

        x_trend_emb = self.embedding(
            torch.cat([x_trend,
                       x_mark[..., 6:7].unsqueeze(2).expand(-1, -1, self.num_nodes, -1),
                       x_mark[..., 2:3].unsqueeze(2).expand(-1, -1, self.num_nodes, -1)], dim=-1)
        ).transpose(1, 2)
        x_season_emb = self.embedding(
            torch.cat([x_season,
                       x_mark[..., 6:7].unsqueeze(2).expand(-1, -1, self.num_nodes, -1),
                       x_mark[..., 2:3].unsqueeze(2).expand(-1, -1, self.num_nodes, -1)], dim=-1)
        ).transpose(1, 2)
        shape = x_trend_emb.shape
        adaptive_emb = self.adaptive_embeddings(
            torch.arange(0, self.seq_len * self.num_nodes).to(x.device)).reshape(1,
                                                                                 self.num_nodes,
                                                                                 self.seq_len,
                                                                                 self.d_model)
        tod_emb = self.tod_embedding((x_mark[..., 6] * self.slice_size_per_day).long()).unsqueeze(1)
        dow_emb = self.dow_embedding(x_mark[..., 2].long()).unsqueeze(1)
        x_emb = torch.concat([x_trend_emb,
                              x_season_emb,
                              adaptive_emb.expand(shape[:-1] + (self.d_model,)),
                              tod_emb.expand(shape[:-1] + (self.d_model,)),
                              dow_emb.expand(shape[:-1] + (self.d_model,)),
                              spatio_emb.expand(shape[:-1] + (self.d_model,))
                              ], dim=-1)
        x_emb = self.emb_projection(x_emb)
        x = x_emb

        for block in self.transformers_list:
            x, _ = block(x)

        x = x.reshape(x.shape[0], self.num_nodes, -1)
        x = self.mix_projection(x)
        x = x.reshape(x.shape[0], self.num_nodes, self.pred_len, self.channel).transpose(1, 2)

        x = self.revin(x, 'denorm', target_slice=slice(0, None))
        if need_transpose:
            x = x.squeeze()
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark=x_mark_enc)
            return dec_out
