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
        x = x.reshape(x.shape[:-2] + (self.channels, ))
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
        x = x.transpose(1,2)
        x = x.reshape(x.shape[:-2] + (self.channels, ))
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

        # [bacth, channel, patch_num, d_model] -> [batch, channel, patch_num, n_heads, d_k] -> [batch, channel, n_heads, patch_num, d_k]
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
            # print(masked_mat)
            attn_score = torch.masked_fill(attn_score, ~masked_mat, -np.inf)

        # print(attn_score[0,0,0])
        output = torch.softmax(attn_score, dim=-1)
        # print(output[0,0,0])
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
        # residual = x
        # x, attn_score = self.masked_attention(x, x, x, masked=True)  # TODO: check if masked_attention is needed
        # x = self.norm1(residual + self.dropout1(x))

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
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

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
    def __init__(self, config, position_embedding_mode: str = 'Traditional'):
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


        # self.emb_dim = self.d_model // 8
        self.embedding = nn.Linear(self.channel + 2, self.d_model)
        self.series_decomposition = SeriesDecomposition(kernel_size=3)
        self.tod_embedding = nn.Embedding(self.slice_size_per_day, self.d_model)
        self.dow_embedding = nn.Embedding(7, self.d_model)
        # self.node_embedding = nn.init.xavier_uniform(
        #     nn.Parameter(torch.empty(config.num_nodes, self.d_model)).to(config.device)
        # )

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
                            # feed_forward=SparseMoE(self.d_model, num_experts=4, top_k=2) if i == config.d_layers - 1 else None
                            feed_forward=SparseMoE(self.d_model, self.d_ff, num_experts=4, top_k=2)
                        ) for i in range(config.d_layers)
                    ])



        self.up_head = nn.Linear(self.d_model, self.channel)


        self.mix_projection = nn.Linear(self.seq_len * self.d_model, self.pred_len * self.channel)


    def forecast(self, x, x_mark=None):
        need_transpose = False
        if len(x.shape) == 3:
            need_transpose=True
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
                              adaptive_emb.expand(shape[:-1] + (self.d_model, )),
                              tod_emb.expand(shape[:-1] + (self.d_model, )),
                              dow_emb.expand(shape[:-1] + (self.d_model, )),
                              spatio_emb.expand(shape[:-1] + (self.d_model, ))
                              ], dim=-1)
        x_emb = self.emb_projection(x_emb)
        x = x_emb

        for block in self.transformers_list:
            x, _ = block(x)

        x = x.reshape(x.shape[0], self.num_nodes, -1)
        x = self.mix_projection(x)
        x = x.reshape(x.shape[0], self.num_nodes, self.pred_len, self.channel).transpose(1,2)

        x = self.revin(x, 'denorm', target_slice=slice(0, None))
        if need_transpose:
            x = x.squeeze()
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark=x_mark_enc)
            return dec_out  # [B, L, D]


if __name__ == '__main__':
    import argparse


    def get_parser():
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
        parser.add_argument('--model', type=str, required=False, default='Autoformer',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=96,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                            default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False,
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true",
                            help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true",
                            help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true",
                            help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true",
                            help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true",
                            help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true",
                            help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

        # TimeXer
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')

        # TimeCMA
        parser.add_argument('--channel', type=int, default=1, help='embedding dimension of TimeCMA')
        parser.add_argument('--res_attention', default=True, help='realformer')
        parser.add_argument('--d_llm', type=int, default=768, help='hidden dimension of LLM')
        parser.add_argument('--num_nodes', type=int, default=7, help='numbers of variate')
        parser.add_argument('--prompt_emb', default=False, help='whether to use pre_prompt embedding')

        args = parser.parse_args()
        if torch.cuda.is_available() and args.use_gpu:
            args.device = torch.device('cuda:{}'.format(args.gpu))
            print('Using GPU')

        return args


    """
    Parameters that will be used in config:
        d_model: 512,
        n_heads: 8,
        dropout: 0.1,
        pred_len: 96,
        patch_len: 16,
        seq_len: 96,
        enc_in: 7,
        d_layers: 6(1)
    """
    # print(a.shape)
    config = get_parser()
    a = torch.rand([32, 96, 7]).to(config.device)
    model = Model(config).to(config.device)

    for name, params in model.named_parameters():
        if params.requires_grad:
            print(params.device)

    output = model(a, a, a, a)
    output.sum().backward()
