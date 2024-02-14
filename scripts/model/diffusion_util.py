import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import math

from timm.models.vision_transformer import Block


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0) # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class TransformerModel(nn.Module):
    def __init__(self, num_pose=34,
                 pose_dim=27, embed_dim=90, hidden_dim=256, depth=4, num_heads=8,
                  decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):

        super().__init__()

        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pose, hidden_dim))

        self.blocks = nn.ModuleList([
            Block(hidden_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(hidden_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(hidden_dim)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim//2, pose_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_emb = time_emb.repeat(1,x.shape[1],1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)

        x = torch.cat([x,ctx_emb], dim=2)

        x = self.linear(x)
        x += self.pos_embedding
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.out(x)
    
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 1000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset = 0):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x + pos_emb
        return x

    def position_encoding(self, offset: int, size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """
        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding
        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset:offset + size]

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_pose=34,
                 pose_dim=27, depth=2, num_heads=4,
                  decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):

        super().__init__()

        self.linear = nn.Linear(pose_dim, pose_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pose, pose_dim))

        self.blocks = nn.ModuleList([
            Block(pose_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(pose_dim)

        self.out = nn.Sequential(
            nn.Linear(pose_dim, pose_dim),
            nn.LeakyReLU(True),
            nn.Linear(pose_dim, pose_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.linear(x)
        if self.pos_embedding.shape[1]!=x.shape[1]:
            x += self.pos_embedding[:,x.shape[1],:]
        else:
            x += self.pos_embedding
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return self.out(x)