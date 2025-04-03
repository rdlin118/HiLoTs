# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn
import numpy as np
from einops import repeat
from collections import Counter

from mmdet3d.models.layers.minkowski_engine_block import (
    IS_MINKOWSKI_ENGINE_AVAILABLE, MinkowskiBasicBlock, MinkowskiBottleneck,
    MinkowskiConvModule)
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        sparseconv_backend (str): Sparse convolutional backend.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backend: str = 'torchsparse',
                 reduce_mode: str = 'random',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        
        self.boundary = nn.Parameter(torch.tensor(0.3))
        self.embed_dim = 256
        self.max_tokens = 1024
        self.attn = CyAttn(voxel_feat_dim=in_channels, 
                           embed_dim=self.embed_dim, 
                           n_enc_layers=6, 
                           n_dec_layers=6, 
                           dropout=0.1,
                           max_tokens=self.max_tokens, 
                           n_heads=1)
        self.close_linear = nn.Linear(self.embed_dim, in_channels)
        self.dist_linear = nn.Linear(self.embed_dim, in_channels)
        
        self.reduce_mode = reduce_mode
        assert self.reduce_mode in ['random', 'density', 'aggregate']
        
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backend} not supported.'
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        if sparseconv_backend == 'torchsparse':
            assert IS_TORCHSPARSE_AVAILABLE, \
                'Please follow `get_started.md` to install Torchsparse.`'
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' \
                else TorchSparseBottleneck
            # for torchsparse, residual branch will be implemented internally
            residual_branch = None
        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available,'
                              'turn to use spconv 1.x in mmcv.')
            input_conv = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' \
                else SparseBottleneck
            residual_branch = partial(
                make_sparse_convmodule,
                conv_type='SubMConv3d',
                order=('conv', 'norm'))
        elif sparseconv_backend == 'minkowski':
            assert IS_MINKOWSKI_ENGINE_AVAILABLE, \
                'Please follow `get_started.md` to install Minkowski Engine.`'
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            decoder_conv = partial(
                MinkowskiConvModule,
                conv_cfg=dict(type='MinkowskiConvNdTranspose'))
            residual_block = MinkowskiBasicBlock if block_type == 'basic' \
                else MinkowskiBottleneck
            residual_branch = partial(MinkowskiConvModule, act_cfg=None)

        self.conv_input = nn.Sequential(
            input_conv(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            input_conv(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    encoder_channels[i],
                    encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(
                                encoder_channels[i],
                                encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            decoder_layer = [
                decoder_conv(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                    indice_key=f'spconv{num_stages-i}')
            ]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] +
                                encoder_channels[-2 - i],
                                decoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i-1}'))
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i-1}'))
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))

    def forward(self, voxel_features: Tensor, coors: Tensor, point2voxel_maps: list) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        batch_size = torch.max(coors[:,-1])
        voxels = []
        attn_matrix = []
        for batch_idx in range(batch_size+1):
            voxel_idx = torch.where(coors[:,-1] == batch_idx)
            
            voxel = voxel_features[voxel_idx]
            
            rho = torch.sqrt(voxel[:,0]**2 + voxel[:,1]**2)
            boundary = torch.quantile(rho.float(), self.boundary, dim=0)
            
            close_inds = torch.where(rho < boundary)
            dist_inds = torch.where(rho >= boundary)
            n_close = len(close_inds[0])
            n_dist = len(dist_inds[0])
            
            if self.reduce_mode == 'random':
                sample_close = torch.tensor(np.random.choice(close_inds[0].cpu(), self.max_tokens))
                sample_dist = torch.tensor(np.random.choice(dist_inds[0].cpu(), 3*self.max_tokens))
            
            elif self.reduce_mode == 'density':
                point2voxel_map = point2voxel_maps[batch_idx]
                
                close_mask = torch.isin(point2voxel_map, close_inds[0])
                point2voxel_close = point2voxel_map[close_mask]
                close_counter = Counter(point2voxel_close.tolist())
                
                density_close_inds = [e[0] for e in close_counter.most_common(self.n_density_close)]
                density_close_mask = torch.isin(point2voxel_close, torch.tensor(density_close_inds).cuda())
                
                density_close = torch.tensor(np.random.choice(point2voxel_close[density_close_mask].cpu(), self.n_density_close))
                random_close_mask = ~density_close_mask
                random_close = torch.tensor(np.random.choice(point2voxel_close[random_close_mask].cpu(), self.n_random_close))
                
                sample_close = torch.cat([density_close, random_close], dim=0)
                
                dist_mask = ~close_mask
                point2voxel_dist = point2voxel_map[dist_mask]
                dist_counter = Counter(point2voxel_dist.tolist())
                
                density_dist_inds = [e[0] for e in dist_counter.most_common(self.n_density_dist)]
                density_dist_mask = torch.isin(point2voxel_dist, torch.tensor(density_dist_inds).cuda())
                density_dist = torch.tensor(np.random.choice(point2voxel_dist[density_dist_mask].cpu(), self.n_density_dist))
                random_dist_mask = ~density_dist_mask
                random_dist = torch.tensor(np.random.choice(point2voxel_dist[random_dist_mask].cpu(), self.n_random_dist))
                
                sample_dist = torch.cat([density_dist, random_dist], dim=0)
                close_voxel_feats = voxel[sample_close]
                dist_voxel_feats = voxel[sample_dist]
            
            elif self.reduce_mode == 'aggregate':
                close_voxel = voxel[close_inds[0]]
                dist_voxel = voxel[dist_inds[0]]
                
                c = n_close % self.max_tokens
                d = n_close // self.max_tokens
                if d == 0:
                    sample_close = torch.tensor(np.random.choice(close_inds[0].cpu(), self.max_tokens))
                    close_voxel_feats = voxel[sample_close]
                else:
                    l = list(range(0, c*(d+1), (d+1)))
                    l.extend(list(range(c*(d+1), n_close, d)))
                    close_voxel_feats = close_voxel[l]
                
                c2 = n_dist % (3*self.max_tokens)
                d2 = n_dist // (3*self.max_tokens)
                if d2 == 0:
                    sample_dist = torch.tensor(np.random.choice(dist_inds[0].cpu(), 3*self.max_tokens))
                    dist_voxel_feats = voxel[sample_dist]
                else:
                    l2 = list(range(0, c2*(d2+1), (d2+1)))
                    l2.extend(list(range(c2*(d2+1), n_dist, d2)))
                    dist_voxel_feats = dist_voxel[l2]
                
                n_close_tokens_to_fuse = close_voxel.shape[0] // self.max_tokens
                close_sets = np.array_split(close_voxel, self.max_tokens)
                close_voxel_feats = []
                for close_set_voxels in close_sets:
                    close_voxel_feats.append(close_set_voxels.mean(dim=0).unsqueeze(0))
                
                dist_sets = np.array_split(dist_voxel, 3*self.max_tokens)
                dist_voxel_feats = []
                for dist_set_voxels in dist_sets:
                    dist_voxel_feats.append(dist_set_voxels.mean(dim=0).unsqueeze(0))
                
                close_voxel_feats = torch.cat(close_voxel_feats, 0)
                dist_voxel_feats = torch.cat(dist_voxel_feats, 0)

            close_voxel_feats = voxel[sample_close]
            dist_voxel_feats = voxel[sample_dist]

            close_attn_vec, dist_attn_vec = self.attn(close_voxel_feats, dist_voxel_feats)
            close_attn_feats = repeat(close_attn_vec, 'd -> n d', n=n_close)
            dist_attn_feats = repeat(dist_attn_vec, 'd -> n d', n=n_dist)
            close_attn_feats = self.close_linear(close_attn_feats)
            dist_attn_feats = self.dist_linear(dist_attn_feats)
            
            attn_feats = torch.zeros([close_attn_feats.shape[0]+dist_attn_feats.shape[0], close_attn_feats.shape[1]], dtype=close_attn_feats.dtype).to('cuda')
            attn_feats[close_inds] = close_attn_feats
            attn_feats[dist_inds] = dist_attn_feats
            attn_matrix.append(attn_feats)
        
        attn_matrix = torch.cat(attn_matrix, dim=0)
        voxel_features += 0*boundary + attn_matrix
        
        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)
        elif self.sparseconv_backend == 'minkowski':
            x = ME.SparseTensor(voxel_features, coors)

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)

            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, laterals[i]))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(
                    x, torch.cat((x.features, laterals[i].features), dim=1))
            elif self.sparseconv_backend == 'minkowski':
                x = ME.cat(x, laterals[i])

            x = decoder_layer[1](x)
            decoder_outs.append(x)

        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features
        else:
            return decoder_outs[-1].F


class CyAttn(BaseModule):
    def __init__(self, voxel_feat_dim, 
                       embed_dim, 
                       n_enc_layers, 
                       n_dec_layers, 
                       dropout=0.1,
                       reduce='average', 
                       max_tokens=1024, 
                       n_heads=8):
        super().__init__()
        self.dropout = dropout
        self.reduce = reduce
        self.max_tokens = max_tokens
        self.norm1 = nn.LayerNorm(voxel_feat_dim)
        self.norm2 = nn.LayerNorm(voxel_feat_dim)
        
        self.pos_embed_close = nn.Embedding(max_tokens, embed_dim)
        self.pos_embed_dist = nn.Embedding(3*max_tokens, embed_dim)
        
        self.input_embedding1 = nn.Linear(voxel_feat_dim, embed_dim)
        self.input_embedding2 = nn.Linear(voxel_feat_dim, embed_dim)
        
        self.voxel_feat_dim = voxel_feat_dim
        self.embed_dim = embed_dim
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        
        self.encoder = nn.ModuleList([
            SelfAttnLayer(embed_dim, embed_dim, n_heads, dropout) for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            CrossAttnLayer(embed_dim, embed_dim, n_heads, dropout) for _ in range(n_dec_layers)
        ])
    
    def forward(self, close, dist):
        pos_close = torch.arange(0, self.max_tokens, 1).to('cuda')
        pos_dist = torch.arange(0, 3*self.max_tokens, 1).to('cuda')
        pos_embed_close = self.pos_embed_close(pos_close)
        pos_embed_dist = self.pos_embed_dist(pos_dist)

        close = self.norm1(close)
        dist = self.norm2(dist)
        
        close = self.input_embedding1(close) + pos_embed_close
        dist = self.input_embedding2(dist) + pos_embed_dist
        
        if self.n_enc_layers == 1:
            attn = self.encoder[0](close)
        elif self.n_enc_layers >= 1:
            attn = self.encoder[0](close)
            for i in range(1, self.n_enc_layers):
                attn = self.encoder[i](attn)
        
        if self.n_dec_layers == 1:
            out = self.decoder[0](attn, dist)
        elif self.n_dec_layers >= 1:
            out = self.decoder[0](attn, dist)
            for i in range(1, self.n_dec_layers):
                out = self.decoder[i](attn, out)
        
        return (attn[0], out[0])  # attn = close, out = dist


class SelfAttnLayer(BaseModule):
    def __init__(self, voxel_feat_dim, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x_clone = x.clone()
        attn = self.norm1(self.attn(query=x, key=x, value=x)[0]) + x_clone
        
        attn_clone = attn.clone()
        out = self.norm2(self.ffn(attn)) + attn_clone
        
        return out


class CrossAttnLayer(BaseModule):
    def __init__(self, voxel_feat_dim, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, close, dist):
        dist_clone = dist.clone()
        attn = self.norm1(self.attn1(query=dist, key=dist, value=dist)[0]) + dist_clone

        attn_clone = attn.clone()
        attn = self.norm2(self.attn2(query=attn, key=close, value=close)[0]) + attn_clone

        attn_clone = attn.clone()
        out = self.norm3(self.ffn(attn)) + attn_clone

        return out
